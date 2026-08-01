import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc, confusion_matrix

def entrenar_y_evaluar_modelos(data_path, output_dir):
    print("Cargando datos limpios...")
    df = pd.read_csv(data_path)

    # Definir variables Y (Dependiente) y X (Explicativas)
    y = df['informal']
    X = df.drop(columns=['informal'])
    X = sm.add_constant(X)  # Agregar constante para la estimación econométrica

    print("Estimando Modelo Logit...")
    logit_mod = sm.Logit(y, X).fit(disp=0)
    
    print("Estimando Modelo Probit...")
    probit_mod = sm.Probit(y, X).fit(disp=0)

    # Predicciones de probabilidad
    pred_logit = logit_mod.predict(X)
    pred_probit = probit_mod.predict(X)

    # Crear carpeta de salidas si no existe
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Generar Curva ROC Comparativa ---
    print("Generando gráfico de Curva ROC...")
    fpr_l, tpr_l, _ = roc_curve(y, pred_logit)
    auc_logit = auc(fpr_l, tpr_l)

    fpr_p, tpr_p, _ = roc_curve(y, pred_probit)
    auc_probit = auc(fpr_p, tpr_p)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_l, tpr_l, color='#2563eb', lw=2, label=f'Logit (AUC = {auc_logit:.3f})')
    plt.plot(fpr_p, tpr_p, color='#dc2626', lw=2, linestyle='--', label=f'Probit (AUC = {auc_probit:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC Comparativa: Logit vs Probit (Informalidad)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Guardar gráfico ROC
    roc_path = os.path.join(output_dir, "curva_roc_comparativa.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- 2. Exportar Tabla Resumen de Métricas ---
    resumen = pd.DataFrame({
        'Modelo': ['Logit', 'Probit'],
        'AIC': [logit_mod.aic, probit_mod.aic],
        'BIC': [logit_mod.bic, probit_mod.bic],
        'Pseudo R2': [logit_mod.prsquared, probit_mod.prsquared],
        'AUC': [auc_logit, auc_probit]
    })
    
    csv_path = os.path.join(output_dir, "resumen_modelos.csv")
    resumen.to_csv(csv_path, index=False)

    print(f"¡Modelado completado!")
    print(f"Gráfico guardado en: {roc_path}")
    print(f"Métricas guardadas en: {csv_path}")

if __name__ == "__main__":
    DATA_CLEAN = "data/enemdu_informalidad_clean.csv"
    OUTPUT_FOLDER = "outputs"
    
    if os.path.exists(DATA_CLEAN):
        entrenar_y_evaluar_modelos(DATA_CLEAN, OUTPUT_FOLDER)
    else:
        print(f"Advertencia: No existe {DATA_CLEAN}. Ejecuta primero src/cleaning.py")
        