# Football Team Dashboard (Streamlit)

Este proyecto es un dashboard interactivo para analizar equipos de una liga y temporada a partir de datos públicos en CSV.

La app permite:
- Calcular tabla de posiciones desde los partidos.
- Ver tablas separadas de rendimiento como local y como visitante.
- Seleccionar un equipo y ver su forma reciente, KPIs y evolución de puntos.

## Fuente de datos
Los CSV se descargan desde Football-Data.co.uk usando el patrón:
https://www.football-data.co.uk/mmz4281/<SEASON>/<DIV>.csv

Ejemplo:
- SEASON = 2324 (2023-24)
- DIV = E0 (Premier League)

## Requisitos
- Python 3.10+ recomendado.

## Ejecución
1) Cree un entorno virtual:
   python -m venv .venv

2) Active el entorno:
   - Windows: .venv\Scripts\activate
   - macOS/Linux: source .venv/bin/activate

3) Instale dependencias:
   pip install -r requirements.txt

4) Corra la app:
   streamlit run app.py

## Notas
- La app cachea los CSV descargados en `data_cache/` para acelerar recargas.
- Si un season code no existe para una liga, la app mostrará un error; pruebe con otro (por ejemplo, 2223, 2324, etc.).
