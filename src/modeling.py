import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc

def entrenar_y_evaluar_modelos(data_path, output_dir):
    print("Cargando datos limpios...")
    df = pd.read_csv(data_path)

    y = df['informal']
    X = df.drop(columns=['informal'])
    X = sm.add_constant(X)

    # Estimación de Modelos
    logit_mod = sm.Logit(y, X).fit(disp=0)
    probit_mod = sm.Probit(y, X).fit(disp=0)

    pred_logit = logit_mod.predict(X)
    pred_probit = probit_mod.predict(X)

    os.makedirs(output_dir, exist_ok=True)

    # Estilo visual de los gráficos (Colores personalizados)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # --- GRAFICO 1: Curva ROC Comparativa ---
    fpr_l, tpr_l, _ = roc_curve(y, pred_logit)
    auc_l = auc(fpr_l, tpr_l)
    fpr_p, tpr_p, _ = roc_curve(y, pred_probit)
    auc_p = auc(fpr_p, tpr_p)

    plt.figure(figsize=(6, 4.5))
    plt.plot(fpr_l, tpr_l, color='#0f172a', lw=2.5, label=f'Logit (AUC = {auc_l:.2f})')
    plt.plot(fpr_p, tpr_p, color='#c2410c', lw=2, linestyle='--', label=f'Probit (AUC = {auc_p:.2f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle=':')
    plt.title('1. Curva ROC Comparativa', fontsize=11, fontweight='bold', color='#0f172a')
    plt.xlabel('1 - Especificidad')
    plt.ylabel('Sensibilidad')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curva_roc_comparativa.png"), dpi=200)
    plt.close()

    # --- GRAFICO 2: Probabilidad Predicha vs. Escolaridad ---
    plt.figure(figsize=(6, 4.5))
    if 'anios_estudio' in df.columns:
        grid_estudio = np.linspace(df['anios_estudio'].min(), df['anios_estudio'].max(), 50)
        # Fijar medias para el resto de variables
        X_mean = X.mean()
        X_pred = pd.DataFrame([X_mean] * 50)
        X_pred['anios_estudio'] = grid_estudio
        
        prob_l_esc = logit_mod.predict(X_pred)
        prob_p_esc = probit_mod.predict(X_pred)

        plt.plot(grid_estudio, prob_l_esc, color='#0f172a', lw=2.5, label='Logit')
        plt.plot(grid_estudio, prob_p_esc, color='#15803d', lw=2, linestyle='--', label='Probit')
        plt.xlabel('Años de Escolaridad')
        plt.ylabel('Probabilidad de Informalidad')
    else:
        plt.text(0.5, 0.5, 'Variable anios_estudio no encontrada', ha='center')

    plt.title('2. Efecto Marginal Escolaridad', fontsize=11, fontweight='bold', color='#0f172a')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prob_vs_escolaridad.png"), dpi=200)
    plt.close()

    # --- GRAFICO 3: Histograma de Probabilidades ---
    plt.figure(figsize=(6, 4.5))
    plt.hist(pred_logit, bins=15, alpha=0.6, color='#0f172a', label='Logit', edgecolor='white')
    plt.hist(pred_probit, bins=15, alpha=0.5, color='#c2410c', label='Probit', edgecolor='white')
    plt.title('3. Distribución de Probabilidades', fontsize=11, fontweight='bold', color='#0f172a')
    plt.xlabel('Probabilidad Predicha (y_hat)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histograma_probabilidades.png"), dpi=200)
    plt.close()

    print("¡Los 3 gráficos han sido generados exitosamente en /outputs!")

if __name__ == "__main__":
    entrenar_y_evaluar_modelos("data/enemdu_informalidad_clean.csv", "outputs")