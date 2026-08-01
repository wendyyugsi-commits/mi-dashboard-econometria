# Registro de Uso de Inteligencia Artificial (AI Agent)

**Proyecto:** Determinantes de la Informalidad Laboral en Ecuador  
**Estudiante:** Wendy Yugsi  
**Modalidad:** Modalidad A (Modelos de Respuesta Binaria: Logit vs. Probit)  
**Herramienta de IA utilizada:** Gemini (Google)  

---

## Declaración de Autoría y Transparencia
El presente proyecto econométrico utilizó herramientas de inteligencia artificial generativa como asistencia técnica para:
1. La estructuración de scripts de Python modularizados (`pandas`, `statsmodels`, `scikit-learn`).
2. La automatización del procesamiento de datos y exportación de métricas a formato `.json`.
3. El diseño de componentes del dashboard interactivo en Vercel (Plotly.js y CSS).
4. La revisión conceptual y de sintaxis para la estimación de los modelos Logit y Probit.

Todos los resultados, códigos y redactados fueron revisados, ejecutados y validados manualmente en el entorno de desarrollo local.

---

## Registro de Prompts Principales

### Prompt 1: Generación y preparación de datos
* **Consulta:** "Necesito un script en Python para estructurar la base de datos ENEMDU del INEC enfocada en informalidad laboral, incluyendo variables explicativas de escolaridad, edad, sexo y estado civil."
* **Uso:** Código utilizado para la limpieza y preparación de variables en Python.

### Prompt 2: Estimación econométrica
* **Consulta:** "Genera un código en Python usando statsmodels para estimar e interpretar un modelo Logit y un Probit con la misma ecuación, obteniendo AIC, BIC, Pseudo R2 y efectos marginales promedio (AME)."
* **Uso:** Estructuración de los modelos y extracción de la tabla de coeficientes econométricos.

### Prompt 3: Evaluación predictiva y Dashboard
* **Consulta:** "Escribe el código para graficar la curva ROC comparativa de Logit vs Probit, la probabilidad predicha según escolaridad y adapútalo para desplegarlo en un dashboard en Vercel."
* **Uso:** Implementación del dashboard interactivo (`index.html` y `style.css`).