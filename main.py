# -*- coding: utf-8 -*-
"""
Desafio de Ciência de Dados: Detecção de Fraude em Cartão de Crédito
--------------------------------------------------------------------
Este script implementa o pipeline completo de Machine Learning para detecção de
fraudes em transações de cartão de crédito, focado em dados altamente desbalanceados.

Requisitos de bibliotecas (instale via terminal se necessário):
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn shap
"""

# %% [0] IMPORTAÇÃO DE BIBLIOTECAS E CONFIGURAÇÕES
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pré-processamento, Divisão e Pipelines
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Modelos de Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# Métricas para Dados Desbalanceados
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score
)

# Técnicas de Reamostragem (Ideias para Evoluir)
try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
    IMBLEARN_DISPONIVEL = True
except ImportError:
    IMBLEARN_DISPONIVEL = False
    print("[AVISO] Biblioteca 'imbalanced-learn' não encontrada. A etapa extra de Under/Oversampling será pulada.")

# Explicabilidade com SHAP
try:
    import shap
    SHAP_DISPONIVEL = True
except ImportError:
    SHAP_DISPONIVEL = False
    print("[AVISO] Biblioteca 'shap' não encontrada. Instale com 'pip install shap' para ver a explicação detalhada.")

# Definição de semente aleatória para reprodutibilidade
RANDOM_STATE = 42


# %% [1] EXPLORAÇÃO E CARREGAMENTO DOS DADOS
def explorar_dados(url: str) -> pd.DataFrame:
    """
    Etapa 1: Carrega o dataset direto da URL com pandas, exibe as primeiras linhas
    e analisa o desbalanceamento com value_counts(normalize=True).
    """
    print("\n" + "=" * 80)
    print("1. EXPLORAÇÃO DOS DADOS")
    print("=" * 80)
    print(f"Baixando e carregando a base de dados diretamente de:\n{url}\nAguarde alguns segundos...")
    
    df = pd.read_csv(url)
    
    print("\n--- Primeiras 5 linhas do Dataset ---")
    print(df.head())
    
    print(f"\nDimensões da base: {df.shape[0]} transações e {df.shape[1]} colunas.")
    
    # Contagem absoluta e proporcional de fraudes
    contagem_absoluta = df['Class'].value_counts()
    proporcao = df['Class'].value_counts(normalize=True)
    
    print("\n--- Proporção da Variável Alvo (Class: 0 = Normal, 1 = Fraude) ---")
    resumo_classes = pd.DataFrame({
        'Quantidade': contagem_absoluta,
        'Proporção': proporcao,
        'Porcentagem (%)': proporcao * 100
    })
    print(resumo_classes)
    
    print("\n[Insight]: Note que ~99.83% das transações são normais e apenas ~0.17% são fraudes.")
    print("Um modelo que chuta '0' para tudo teria 99.83% de acurácia, mas 0% de Recall na classe de fraude!")
    
    return df


# %% [2] PREPARAÇÃO E ENGENHARIA DE VARIÁVEIS
def preparar_dados(df: pd.DataFrame):
    """
    Etapa 2 + Ideia de Evolução:
    - Cria o log do valor da transação com np.log1p(Amount).
    - Cria variáveis de comportamento temporal (hora do dia e diferença de tempo).
    - Separa treino e teste usando stratify=y para preservar os 0,17% de fraude.
    - Padroniza os dados com StandardScaler (ajustado apenas no treino para evitar Data Leakage).
    """
    print("\n" + "=" * 80)
    print("2. PREPARAÇÃO DOS DADOS E ENGENHARIA DE VARIÁVEIS")
    print("=" * 80)
    
    dados = df.copy()
    
    # 1. Transformação logarítmica no valor da transação (reduz assimetria de valores muito altos)
    dados['Amount_Log'] = np.log1p(dados['Amount'])
    
    # 2. Ideia para Evoluir: Variáveis de comportamento ao longo do tempo
    # Converte os segundos acumulados em hora relativa do dia (ciclo de 24h)
    dados['Hour_of_Day'] = (dados['Time'] / 3600) % 24
    # Calcula o intervalo de tempo desde a transação imediatamente anterior no sistema
    dados['Time_Diff'] = dados['Time'].diff().fillna(0)
    
    print("Variáveis criadas com sucesso:")
    print(" - 'Amount_Log': Logaritmo natural de (Amount + 1) usando np.log1p")
    print(" - 'Hour_of_Day' e 'Time_Diff': Variáveis comportamentais de sequência no tempo")
    
    # Separação entre variáveis explicativas (X) e variável alvo (y)
    X = dados.drop(columns=['Class'])
    y = dados['Class']
    
    # Divisão Treino e Teste estratificada (80% treino, 20% teste)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\nDivisão Treino/Teste concluída com stratify=y:")
    print(f" - Treino: {X_train_raw.shape[0]} amostras | Taxa de Fraude: {y_train.mean()*100:.4f}%")
    print(f" - Teste : {X_test_raw.shape[0]} amostras  | Taxa de Fraude: {y_test.mean()*100:.4f}%")
    
    # Padronização com StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw),
        columns=X.columns,
        index=X_train_raw.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw),
        columns=X.columns,
        index=X_test_raw.index
    )
    
    print("Padronização com StandardScaler concluída!")
    return X_train_raw, X_test_raw, X_train_scaled, X_test_scaled, y_train, y_test


# %% [3] e [4] TREINAMENTO E AVALIAÇÃO DE MODELOS
def avaliar_modelo(nome: str, modelo, X_test, y_test, limiar: float = 0.5):
    """
    Calcula probabilidades, aplica o limiar de decisão (threshold), exibe o
    relatório de classificação e retorna um dicionário com as métricas e curvas.
    """
    # Obtém a probabilidade prevista para a classe positiva (1 = Fraude)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    
    # Aplica o limiar de decisão (padrão 0.5 ou ajustado, ex: 0.3)
    y_pred = (y_prob >= limiar).astype(int)
    
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n>>> Resultados: {nome} (Limiar = {limiar}) <<<")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraude (1)'], digits=4))
    print(f"Matriz de Confusão:\n{cm}")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC (Precision-Recall AUC): {pr_auc:.4f}")
    
    return {
        'Modelo': nome,
        'Limiar': limiar,
        'Recall (Fraude)': rec,
        'Precisão (Fraude)': prec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'y_prob': y_prob
    }


def plotar_curvas_avaliacao(resultados_lista: list, y_test):
    """
    Etapa 4: Plota lado a lado a Curva ROC e a Curva de Precisão-Recall (a mais importante
    para problemas desbalanceados) e salva em arquivo PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Curva ROC
    for res in resultados_lista:
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        axes[0].plot(fpr, tpr, lw=2, label=f"{res['Modelo']} (AUC = {res['ROC-AUC']:.4f})")
    
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Aleatório')
    axes[0].set_title('Curva ROC (Receiver Operating Characteristic)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Taxa de Falsos Positivos (FPR)')
    axes[0].set_ylabel('Taxa de Verdadeiros Positivos (Recall)')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    
    # 2. Curva Precision-Recall (Mais útil em dados desbalanceados)
    for res in resultados_lista:
        precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
        axes[1].plot(recall, precision, lw=2, label=f"{res['Modelo']} (PR-AUC = {res['PR-AUC']:.4f})")
        
    axes[1].set_title('Curva de Precisão e Recall (Foco em Fraude)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Recall (Fraudes Encontradas)')
    axes[1].set_ylabel('Precisão (Acertos entre Alertas de Fraude)')
    axes[1].legend(loc='lower left')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('curvas_roc_e_precision_recall.png', dpi=150)
    print("\n[Gráfico Salvo]: 'curvas_roc_e_precision_recall.png'")
    plt.close()


def treinar_e_comparar_modelos(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Etapa 3 e 4: Treina Regressão Logística (Baseline), Random Forest (com class_weight='balanced'),
    XGBoost e ExtraTrees, comparando todas as métricas na classe de fraude.
    """
    print("\n" + "=" * 80)
    print("3 & 4. TREINAMENTO E AVALIAÇÃO DOS MODELOS PRINCIPAIS")
    print("=" * 80)
    
    # Cálculo de peso proporcional para o XGBoost (Total Normais / Total Fraudes)
    proporcao_neg_pos = (y_train == 0).sum() / (y_train == 1).sum()
    
    modelos = {
        '1. Regressão Logística (Baseline)': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        '2. Regressão Logística (Balanced)': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        '3. Random Forest (Balanced)': RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        '4. XGBoost (Scale Pos Weight)': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=proporcao_neg_pos,
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        '5. Extra Trees (Balanced - Extra)': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    modelos_treinados = {}
    resultados_avaliacao = []
    
    for nome, mod in modelos.items():
        print(f"\nTreinando {nome}...")
        mod.fit(X_train_scaled, y_train)
        modelos_treinados[nome] = mod
        res = avaliar_modelo(nome, mod, X_test_scaled, y_test, limiar=0.5)
        resultados_avaliacao.append(res)
        
    # Plota as curvas ROC e Precision-Recall para comparação visual
    plotar_curvas_avaliacao(resultados_avaliacao, y_test)
    
    return modelos_treinados, resultados_avaliacao


# %% [IDEIA PARA EVOLUIR] UNDERSAMPLING VS OVERSAMPLING (SMOTE)
def comparar_reamostragem(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Ideia para Evoluir: Testa na prática Undersampling vs Oversampling (SMOTE)
    e compara o Recall obtido na base de teste.
    """
    if not IMBLEARN_DISPONIVEL:
        return []
        
    print("\n" + "=" * 80)
    print("EVOLUÇÃO: COMPARAÇÃO ENTRE UNDERSAMPLING E OVERSAMPLING (SMOTE)")
    print("=" * 80)
    
    resultados_reamostragem = []
    
    # 1. Undersampling (Reduz a classe majoritária para equilibrar)
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)
    X_under, y_under = rus.fit_resample(X_train_scaled, y_train)
    print(f"Undersampling concluído: {len(y_under)} amostras de treino ({y_under.sum()} fraudes).")
    
    rf_under = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf_under.fit(X_under, y_under)
    res_under = avaliar_modelo("Random Forest + Undersampling", rf_under, X_test_scaled, y_test, limiar=0.5)
    resultados_reamostragem.append(res_under)
    
    # 2. Oversampling com SMOTE (Cria exemplos sintéticos da classe de fraude)
    smote = SMOTE(sampling_strategy=0.2, random_state=RANDOM_STATE)
    X_over, y_over = smote.fit_resample(X_train_scaled, y_train)
    print(f"\nOversampling (SMOTE) concluído: {len(y_over)} amostras de treino ({y_over.sum()} fraudes).")
    
    rf_over = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf_over.fit(X_over, y_over)
    res_over = avaliar_modelo("Random Forest + Oversampling (SMOTE)", rf_over, X_test_scaled, y_test, limiar=0.5)
    resultados_reamostragem.append(res_over)
    
    return resultados_reamostragem


# %% [5] AJUSTE DE PIPELINE, LIMIAR DE DECISÃO (0.5 -> 0.3) E GRIDSEARCHCV
def ajustar_pipeline_e_limiar(X_train_raw, X_test_raw, y_train, y_test):
    """
    Etapa 5 + Ideia para Evoluir:
    - Une StandardScaler e o classificador dentro de um Pipeline do Scikit-Learn.
    - Baixa o limiar de decisão de 0.5 para 0.3 para capturar mais fraudes (aumentar Recall).
    - Executa GridSearchCV escolhendo os melhores hiperparâmetros pela métrica 'recall'.
    """
    print("\n" + "=" * 80)
    print("5. AJUSTE: PIPELINE + LIMIAR DE DECISÃO (0.3) + GRIDSEARCHCV (FOCO EM RECALL)")
    print("=" * 80)
    
    # Construção do Pipeline integrando Padronização e Random Forest
    pipeline_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
    ])
    
    # Grade de hiperparâmetros para busca
    param_grid = {
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [6, 10, 14],
        'rf__min_samples_leaf': [2, 5]
    }
    
    cv_estratificado = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    print("Executando GridSearchCV otimizando diretamente pelo RECALL...")
    grid_search = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid,
        scoring='recall',
        cv=cv_estratificado,
        n_jobs=-1,
        verbose=1
    )
    
    # Treina passando os dados brutos (o Pipeline aplica o StandardScaler internamente sem vazamento)
    grid_search.fit(X_train_raw, y_train)
    
    print(f"\nMelhores Hiperparâmetros encontrados: {grid_search.best_params_}")
    print(f"Melhor Recall médio na Validação Cruzada: {grid_search.best_score_:.4f}")
    
    melhor_pipeline = grid_search.best_estimator_
    
    # Comparando o impacto de reduzir o limiar de decisão de 0.5 para 0.3 no mesmo Pipeline
    print("\n--- Comparando Limiar Padrão (0.5) vs Limiar Ajustado (0.3) no Pipeline ---")
    res_05 = avaliar_modelo("Pipeline RF (GridSearch)", melhor_pipeline, X_test_raw, y_test, limiar=0.5)
    res_03 = avaliar_modelo("Pipeline RF (GridSearch + Limiar 0.3)", melhor_pipeline, X_test_raw, y_test, limiar=0.3)
    
    ganho_recall = (res_03['Recall (Fraude)'] - res_05['Recall (Fraude)']) * 100
    print(f"\n[Conclusão do Ajuste de Limiar]: Ao reduzir o limiar de 0.5 para 0.3, o Recall na classe")
    print(f"de fraude variou em {ganho_recall:+.2f} pontos percentuais, capturando transações suspeitas")
    print(f"que antes passariam despercebidas!")
    
    return melhor_pipeline, [res_05, res_03]


# %% [6] EXPLICABILIDADE DO MODELO (IMPORTÂNCIA DAS VARIÁVEIS E SHAP)
def explicar_modelo(modelo_arvore, X_test_scaled, y_test):
    """
    Etapa 6:
    - Extrai e plota as variáveis mais importantes do modelo.
    - Usa o SHAP (SHapley Additive exPlanations) para explicar globalmente o modelo
      e detalhar exatamente o que levou uma transação real de fraude a ser marcada.
    """
    print("\n" + "=" * 80)
    print("6. EXPLICABILIDADE: IMPORTÂNCIA DAS VARIÁVEIS E ANÁLISE SHAP")
    print("=" * 80)
    
    # 1. Importância nativa das variáveis (Feature Importances)
    importancias = pd.Series(
        modelo_arvore.feature_importances_,
        index=X_test_scaled.columns
    ).sort_values(ascending=False)
    
    print("\n--- Top 10 Variáveis Mais Importantes para Detectar Fraude ---")
    print(importancias.head(10))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias.head(12).values, y=importancias.head(12).index, palette='viridis')
    plt.title('Top 12 Variáveis Mais Importantes (Random Forest)', fontsize=13, fontweight='bold')
    plt.xlabel('Importância Relativa')
    plt.ylabel('Variável')
    plt.tight_layout()
    plt.savefig('importancia_variaveis.png', dpi=150)
    print("[Gráfico Salvo]: 'importancia_variaveis.png'")
    plt.close()
    
    # 2. Análise SHAP para explicar decisões individuais e globais
    if not SHAP_DISPONIVEL:
        print("\n[Pulado]: Instale a biblioteca 'shap' para gerar os gráficos e explicações SHAP.")
        return
        
    print("\nCalculando valores SHAP para explicar as decisões do modelo...")
    
    # Seleciona uma amostra de transações normais e de fraudes reais para agilizar o cálculo SHAP
    indices_fraude = y_test[y_test == 1].index[:30]
    indices_normais = y_test[y_test == 0].index[:120]
    amostra_indices = indices_fraude.union(indices_normais)
    X_amostra_shap = X_test_scaled.loc[amostra_indices]
    
    explainer = shap.TreeExplainer(modelo_arvore)
    shap_values = explainer(X_amostra_shap)
    
    # Em classificadores binários (RandomForest), selecionamos a dimensão da classe 1 (Fraude)
    if len(shap_values.shape) == 3:
        shap_values_fraude = shap_values[:, :, 1]
    else:
        shap_values_fraude = shap_values
        
    # Gráfico Global SHAP Summary Plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values_fraude, X_amostra_shap, show=False)
    plt.title('Impacto Global das Variáveis na Probabilidade de Fraude (SHAP)', fontsize=12)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=150)
    print("[Gráfico Salvo]: 'shap_summary_plot.png'")
    plt.close()
    
    # Explicação Local: O que levou uma única transação fraudulenta a ser marcada como fraude?
    id_transacao_fraude = indices_fraude[0]
    posicao_na_amostra = list(X_amostra_shap.index).index(id_transacao_fraude)
    
    valores_transacao = X_amostra_shap.loc[id_transacao_fraude]
    pesos_shap_transacao = pd.Series(
        shap_values_fraude[posicao_na_amostra].values,
        index=X_amostra_shap.columns
    )
    
    # Ordena as variáveis que mais empurraram a decisão para FRAUDE (valores SHAP positivos mais altos)
    principais_motivos = pd.DataFrame({
        'Valor Padronizado da Variável': valores_transacao,
        'Impacto SHAP (+ empurra p/ Fraude)': pesos_shap_transacao
    }).sort_values(by='Impacto SHAP (+ empurra p/ Fraude)', ascending=False)
    
    prob_prevista = modelo_arvore.predict_proba(X_amostra_shap.loc[[id_transacao_fraude]])[0, 1]
    
    print(f"\n--- Explicação Individual (SHAP) para a Transação ID #{id_transacao_fraude} (Fraude Real) ---")
    print(f"Probabilidade de Fraude calculada pelo modelo: {prob_prevista*100:.2f}%")
    print("\nTop 5 variáveis que mais pesaram para marcar esta transação como FRAUDE:")
    print(principais_motivos.head(5))


# %% [EXECUÇÃO PRINCIPAL DO PIPELINE]
def main():
    URL_DADOS = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    
    # 1. Explorar
    df = explorar_dados(URL_DADOS)
    
    # 2. Preparar
    X_train_raw, X_test_raw, X_train_scaled, X_test_scaled, y_train, y_test = preparar_dados(df)
    
    # 3 e 4. Treinar e Avaliar Modelos Base
    modelos_treinados, resultados_base = treinar_e_comparar_modelos(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Evolução: Comparar Undersampling e Oversampling
    resultados_reamostragem = comparar_reamostragem(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # 5. Ajustar Pipeline, Limiar (0.3) e GridSearchCV focado em Recall
    melhor_pipeline, resultados_pipeline = ajustar_pipeline_e_limiar(
        X_train_raw, X_test_raw, y_train, y_test
    )
    
    # 6. Explicar com Feature Importance e SHAP (usando o Random Forest ou XGBoost)
    modelo_para_explicar = modelos_treinados['3. Random Forest (Balanced)']
    explicar_modelo(modelo_para_explicar, X_test_scaled, y_test)
    
    # Tabela Final Consolidada ordenada pelo Recall na classe de fraude
    print("\n" + "=" * 80)
    print("RESUMO FINAL: COMPARAÇÃO DE TODOS OS MODELOS (ORDENADO PELO RECALL)")
    print("=" * 80)
    todos_resultados = resultados_base + resultados_reamostragem + resultados_pipeline
    df_comparacao = pd.DataFrame(todos_resultados).drop(columns=['y_prob'])
    df_comparacao = df_comparacao.sort_values(by='Recall (Fraude)', ascending=False).reset_index(drop=True)
    print(df_comparacao.to_string(index=False))
    print("\nPipeline finalizado com sucesso!")


if __name__ == "__main__":
    main()