# Clasificación asistida de PQRS: código y evidencias agregadas

Material complementario de la tesis de maestría *Clasificación asistida de PQRS mediante modelos de lenguaje: apoyo al enrutamiento inicial con validación humana* (Maestría en Inteligencia Artificial, Universidad de La Salle, 2026), de Juan Pablo Garzón Dueñas, Rodolfo Rodríguez Sarmiento y Alexander David Vargas León.

Este repositorio (https://github.com/juanpablogd/clasificacion-asistida-pqrs) permite verificar las cifras de las búsquedas de hiperparámetros, el ensamble, la calibración y la cobertura, y consultar el código de los análisis. No contiene textos de solicitudes, radicados, anotaciones individuales, predicciones por caso ni pesos de modelos.

## Datos de partida

Conjunto público `Registro_PQRS_Mindeportes` del Ministerio del Deporte, Portal Nacional de Datos Abiertos de Colombia (https://www.datos.gov.co/d/bx26-pux4), con licencia CC BY-SA 4.0. Los datos no se redistribuyen aquí.

## Correspondencia con la tesis

- Rejillas y resultados del barrido clásico (capítulo de resultados y anexo C): `metadatos/barrido_clasico_metadata.json` y `metadatos/barrido_rf_metadata.json`.
- Saturación de la búsqueda ampliada (figura y anexo C): `metadatos/optuna_saturacion_metadata.json`.
- Barrido de pesos del ensamble (figura y anexo D): `metadatos/ensamble_exploratorio_metadata.json`; selección en validación: `metadatos/ensamble_validacion_metadata.json`.
- Diagrama de fiabilidad y cobertura contra precisión: `metadatos/calibracion_metadata.json`, `tablas/fiabilidad_bins*.csv` y `tablas/cobertura_precision.csv`.
- Distribución de las 42 dependencias en la prueba: `tablas/pr_por_clase.csv`.
- Comprobación del contenido de la partición de prueba: `codigo/Proyecto/Scripts/verificar_contenido_particion.py`.
- Acierto según la longitud del asunto y el tipo de solicitud (anexo B): `codigo/Proyecto/Scripts/analisis_longitud_y_tipo.py`.

## Contenido y procedencia

- `metadatos/barrido_clasico_metadata.json` y `barrido_rf_metadata.json`: configuraciones, 20 y 5 puntajes de validación conservados, métricas de prueba y controles agregados. Proceden de `Proyecto/artifacts/barrido_clasico/`.
- `metadatos/optuna_saturacion_metadata.json`: espacio, mejor configuración, importancias, curva y los 100 puntajes en validación. Procede de `Proyecto/artifacts/optuna_saturacion/`. No se incluye la base SQLite.
- `metadatos/ensamble_exploratorio_metadata.json`: barrido de siete pesos más promedio de rangos, cota oráculo y pruebas históricas elegidas sobre el propio test. Es exploratorio y tiene sesgo de selección.
- `metadatos/ensamble_validacion_metadata.json`: trece alternativas evaluadas en validación, peso elegido y evaluación en prueba. Ambos archivos de ensamble proceden de `Proyecto/artifacts/ensamble/` y permanecen separados para no confundir los diseños.
- `metadatos/calibracion_metadata.json` y `tablas/fiabilidad_bins*.csv`, `tablas/cobertura_precision.csv`: agregados de calibración y cobertura de `Proyecto/artifacts/calibracion/`.
- `metadatos/revision_general_20261124.json`: comprobación independiente de métricas, tres casos compartidos entre brazos del piloto y sensibilidad al excluirlos, y diagnóstico de la consolidación global anterior al corte temporal. Su script está en `codigo/Proyecto/Entregables/09_Revision_Tutor_20261124/`.
- `tablas/pr_por_clase.csv`: soporte, precisión promedio y F1 por dependencia, de `Proyecto/artifacts/cierre_baseline/`. Los nombres son categorías institucionales; no hay etiquetas ni textos por solicitud.
- `codigo/Proyecto/`: copias exactas de los scripts indicados en el manifiesto, con su ubicación relativa dentro del checkout. Se entregan para inspección del procedimiento.
- `codigo/Busqueda_historica_fragmento.py`: código literal de la celda de búsqueda aleatoria del cuaderno, sin sus salidas ni datos. El encabezado enumera las dependencias del estado previo.
- `MANIFIESTO_SHA256.json`: tamaño, huella SHA-256, fuente y transformación de cada archivo. Excluye su propia huella para evitar autorreferencia; su integridad también se comprueba al verificar el ZIP.

Los JSON históricos son selecciones curadas de los originales: se conservaron las cifras y se omitieron narrativas obsoletas. La auditoría general posterior se identifica por separado: recalcula métricas sobre predicciones conservadas y comprueba agregados, sin entrenar modelos. El empaquetado no ejecuta esos análisis. Cada fuente y cada copia exacta quedan identificadas por SHA-256.

## Interpretación vigente y límites

No se conserva `cv_results_` ni los doce puntajes de validación cruzada de la búsqueda aleatoria inicial; no se reconstruyen ni se sustituyen por ejecuciones nuevas. El espacio y el ganador están en el fragmento del cuaderno; los resultados agregados históricos de la prueba de 8.657 registros se documentan en la tesis. En SVD, la rejilla define dos variantes de 300 componentes cuyos puntajes no aparecen en los metadatos; se conserva el resultado de 600 componentes. Los 100 ensayos Optuna usan SGD con pérdida logística: sus valores absolutos no son métricas del artefacto final `lbfgs`.

Los scripts se mantienen exactos para permitir su contraste con la ejecución histórica. Algunos comentarios y mensajes contienen formulaciones anteriores, sustituidas por la tesis revisada: ausencia de diferencia significativa no demuestra equivalencia; la GPU fue la infraestructura utilizada para el codificador y no una necesidad intrínseca de toda inferencia; el top-3 del ensamble no tiene prueba pareada propia. Frente al lineal, el ensamble con peso elegido en validación no muestra mejora significativa de F1 macro (IC que cruza cero) y sí de top-1 (McNemar). Los JSON curados y la tesis reflejan esta interpretación; los comentarios históricos del código no la reemplazan.

El oráculo solo acota la selección entre las dos predicciones top-1: no cualquier ensamble de probabilidades o rangos. La consolidación global permite que información de 2021 influya en etiquetas atribuidas a 2020; las métricas temporales conservadas no cuantifican ese efecto. Las muestras humanas tienen tres casos compartidos; su exclusión mantiene el patrón de desacuerdo, sin controlar orden ni experiencia.

Las cifras de confianza, cobertura y precisión son retrospectivas frente a etiquetas históricas. La calibración se analizó después del piloto y no se aplicó a la aplicación usada allí. No prueban ahorro temporal, efecto causal, despliegue institucional ni enrutamiento autónomo. Toda sugerencia conserva revisión humana.

## Reproducción y uso del código

El contenido se reunió con un script de empaquetado del proyecto original que solo utiliza la biblioteca estándar de Python, no entrena ni hace inferencia, y registra en el manifiesto la fuente y la huella de cada archivo.

Para ejecutar los experimentos originales se requiere un checkout con la estructura `Proyecto/`, las dependencias del proyecto y los datos, artefactos y pesos originales, que no forman parte de este paquete. Entre las bibliotecas empleadas figuran NumPy, pandas, SciPy, scikit-learn, joblib, Optuna y, para el codificador, PyTorch y Transformers; Parquet requiere un motor compatible. Las versiones históricas figuran en los metadatos y en la documentación del proyecto. No se verificó aquí un entorno de entrenamiento nuevo.

Los scripts resuelven rutas relativas al checkout y pueden escribir resultados. Optuna reutiliza la caché creada por el barrido; los ensambles requieren pesos y particiones conservadas; la calibración requiere probabilidades históricas y el artefacto lineal. El fragmento histórico necesita las variables de las celdas previas y no es un programa autónomo. No basta clonar este repositorio para ejecutar los experimentos.

Se excluyeron por lista explícita de archivos los textos de solicitudes, radicados, anotaciones, predicciones individuales, bases de datos, pesos, credenciales y archivos de datos. El repositorio incluye solo tablas agregadas, metadatos curados y código fuente.

## Licencia

El código (`codigo/`) se distribuye bajo licencia MIT (archivo `LICENSE`). Las tablas y los metadatos agregados (`tablas/` y `metadatos/`) se derivan del conjunto `Registro_PQRS_Mindeportes` y se distribuyen bajo la licencia CC BY-SA 4.0 de la fuente.

## Cómo citar

Juan Pablo Garzón Dueñas, Rodolfo Rodríguez Sarmiento, Alexander David Vargas León (2026). *Clasificación asistida de PQRS mediante modelos de lenguaje: apoyo al enrutamiento inicial con validación humana*. Tesis de maestría, Maestría en Inteligencia Artificial, Universidad de La Salle, Bogotá.
