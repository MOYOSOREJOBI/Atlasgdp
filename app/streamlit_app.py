from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from atlas_gdp.config import load_settings
from atlas_gdp.pipeline.run import (
    load_forecast_payloads_from_manifest,
    load_latest_manifest,
    manifest_country_list,
    run_pipeline,
)


st.set_page_config(page_title="ATLAS-GDP Simulator", page_icon="📈", layout="wide")

ART_DIR = ROOT / "artifacts" / "latest"
LANGUAGES = ["English", "Français", "Português", "Español", "日本語", "العربية"]
LANG_CODES = {
    "English": "en",
    "Français": "fr",
    "Português": "pt",
    "Español": "es",
    "日本語": "ja",
    "العربية": "ar",
}

TEXT = {
    "en": {
        "title": "ATLAS-GDP Simulator",
        "subtitle": "Mixed-frequency GDP nowcasts, recursive multi-horizon forecasts, scenario stress testing, and audit-grade traceability.",
        "sidebar": "Institutional GDP forecasting dashboard",
        "country": "Country",
        "language": "Language",
        "as_of": "As-of date",
        "max_h": "Max horizon (quarters)",
        "scenario": "Scenario simulator",
        "tighten": "Financial tightening",
        "commodity": "Commodity shock",
        "demand": "Demand boost",
        "refresh": "Refresh artifacts",
        "nowcast": "Nowcast mean",
        "interval50": "50% interval",
        "interval90": "90% interval",
        "shock": "Shock probability",
        "forecast_tab": "Forecast",
        "sim_tab": "Simulator",
        "map_tab": "World Map",
        "backtest_tab": "Backtest",
        "audit_tab": "Audit",
        "ask_tab": "Ask ATLAS",
        "about_tab": "About",
        "blend": "Model blend",
        "drivers": "Drivers",
        "driver_chart": "Current driver decomposition",
        "backtest": "Backtest summary",
        "lineage": "Data lineage",
        "forecast_artifact": "Forecast artifact",
        "drivers_artifact": "Drivers artifact",
        "world_title": "Latest global snapshot",
        "top_countries": "Highest risk countries",
        "ask_caption": "Grounded explainer based on current artifacts.",
        "ask_placeholder": "Ask a macro question about the current forecast",
        "about_header": "What this system does",
        "about_body": "ATLAS-GDP combines a baseline panel model, a bridge model, a dynamic factor model, MIDAS, a small BVAR, and a stacked ensemble. It nowcasts current-quarter growth and extends that path recursively for future quarters. Each forecast includes uncertainty bands, a cross-model weight view, and data lineage so you can audit where the number came from.",
        "about_methods": "Method notes",
        "about_methods_body": "The recursive path is generated from the trained bundle. Horizon 0 uses the current engineered macro state. Each later horizon feeds the previous forecast back into lagged GDP features, keeps slow-moving structural variables persistent, and mean-reverts inflation and unemployment toward recent country anchors. This is a lightweight but coherent forecasting simulator, not a full production macro desk system.",
        "about_limits": "Limitations",
        "about_limits_body": "Offline mode uses a compact sample panel. Data revisions are not modeled. Quarterly labels are simulated from annual sample rows when high-frequency releases are unavailable. Scenario sliders apply transparent stress adjustments on top of the recursive baseline path, so they are useful for sensitivity analysis rather than causal policy estimation.",
        "disclaimer": "Not financial advice. Forecasts depend on data quality, release timing, and model assumptions.",
        "sim_impact": "Scenario impact",
        "sim_prob": "Simulated probability table",
        "map_metric": "Map metric",
        "metric_shock": "Shock probability",
        "metric_growth": "Nowcast mean",
        "question_recession": "What is the recession risk here?",
        "question_drivers": "What is driving the forecast?",
        "question_models": "How are the models weighted?",
    },
    "fr": {
        "title": "Simulateur ATLAS-GDP",
        "subtitle": "Nowcasts du PIB à fréquence mixte, prévisions récursives multi-horizons, stress tests de scénarios et traçabilité d’audit.",
        "sidebar": "Tableau de bord institutionnel de prévision du PIB",
        "country": "Pays",
        "language": "Langue",
        "as_of": "Date d’arrêté",
        "max_h": "Horizon maximal (trimestres)",
        "scenario": "Simulateur de scénario",
        "tighten": "Durcissement financier",
        "commodity": "Choc sur les matières premières",
        "demand": "Relance de la demande",
        "refresh": "Rafraîchir les artefacts",
        "nowcast": "Nowcast moyen",
        "interval50": "Intervalle 50 %",
        "interval90": "Intervalle 90 %",
        "shock": "Probabilité de choc",
        "forecast_tab": "Prévision",
        "sim_tab": "Simulation",
        "map_tab": "Carte mondiale",
        "backtest_tab": "Backtest",
        "audit_tab": "Audit",
        "ask_tab": "Demander à ATLAS",
        "about_tab": "À propos",
        "blend": "Pondération des modèles",
        "drivers": "Facteurs moteurs",
        "driver_chart": "Décomposition des facteurs actuels",
        "backtest": "Résumé du backtest",
        "lineage": "Traçabilité des données",
        "forecast_artifact": "Artefact de prévision",
        "drivers_artifact": "Artefact des facteurs",
        "world_title": "Vue mondiale actuelle",
        "top_countries": "Pays les plus risqués",
        "ask_caption": "Explicateur fondé sur les artefacts actuels.",
        "ask_placeholder": "Posez une question macro sur la prévision actuelle",
        "about_header": "Ce que fait ce système",
        "about_body": "ATLAS-GDP combine un modèle panel de base, un modèle bridge, un modèle à facteurs dynamiques, MIDAS, un petit BVAR et un ensemble empilé. Il nowcaste la croissance courante puis prolonge ce chemin de façon récursive pour les trimestres suivants.",
        "about_methods": "Notes méthodologiques",
        "about_methods_body": "L’horizon 0 utilise l’état macro courant. Chaque horizon suivant réinjecte la prévision précédente dans les retards du PIB, maintient les variables structurelles lentes et fait revenir inflation et chômage vers des ancrages récents du pays.",
        "about_limits": "Limites",
        "about_limits_body": "Le mode hors ligne utilise un petit panel d’exemple. Les révisions de données ne sont pas modélisées. Les trimestres sont simulés à partir de lignes annuelles quand les sorties haute fréquence manquent.",
        "disclaimer": "Pas un conseil financier. Les prévisions dépendent de la qualité des données et des hypothèses du modèle.",
        "sim_impact": "Impact du scénario",
        "sim_prob": "Table des probabilités simulées",
        "map_metric": "Mesure cartographique",
        "metric_shock": "Probabilité de choc",
        "metric_growth": "Nowcast moyen",
        "question_recession": "Quel est le risque de récession ici ?",
        "question_drivers": "Quels facteurs expliquent la prévision ?",
        "question_models": "Comment les modèles sont-ils pondérés ?",
    },
    "pt": {
        "title": "Simulador ATLAS-GDP",
        "subtitle": "Nowcasts de PIB em frequência mista, previsões recursivas multi-horizonte, testes de estresse e rastreabilidade.",
        "sidebar": "Painel institucional de previsão do PIB",
        "country": "País",
        "language": "Idioma",
        "as_of": "Data base",
        "max_h": "Horizonte máximo (trimestres)",
        "scenario": "Simulador de cenário",
        "tighten": "Aperto financeiro",
        "commodity": "Choque de commodities",
        "demand": "Impulso de demanda",
        "refresh": "Atualizar artefatos",
        "nowcast": "Nowcast médio",
        "interval50": "Intervalo de 50%",
        "interval90": "Intervalo de 90%",
        "shock": "Probabilidade de choque",
        "forecast_tab": "Previsão",
        "sim_tab": "Simulação",
        "map_tab": "Mapa global",
        "backtest_tab": "Backtest",
        "audit_tab": "Auditoria",
        "ask_tab": "Perguntar ao ATLAS",
        "about_tab": "Sobre",
        "blend": "Mistura de modelos",
        "drivers": "Vetores de impulso",
        "driver_chart": "Decomposição dos vetores atuais",
        "backtest": "Resumo do backtest",
        "lineage": "Linhagem dos dados",
        "forecast_artifact": "Artefato de previsão",
        "drivers_artifact": "Artefato dos vetores",
        "world_title": "Panorama global atual",
        "top_countries": "Países com maior risco",
        "ask_caption": "Explicador baseado nos artefatos atuais.",
        "ask_placeholder": "Faça uma pergunta macro sobre a previsão atual",
        "about_header": "O que este sistema faz",
        "about_body": "ATLAS-GDP combina modelo em painel, bridge, fator dinâmico, MIDAS, um BVAR pequeno e um ensemble empilhado. Ele produz nowcast do trimestre corrente e estende a trajetória de forma recursiva.",
        "about_methods": "Notas metodológicas",
        "about_methods_body": "O horizonte 0 usa o estado macro corrente. Os horizontes seguintes alimentam a previsão anterior nas defasagens de PIB, mantêm variáveis estruturais estáveis e aproximam inflação e desemprego das âncoras recentes do país.",
        "about_limits": "Limitações",
        "about_limits_body": "O modo offline usa um painel amostral pequeno. Revisões de dados não são modeladas. Os trimestres são simulados a partir de linhas anuais quando faltam liberações de alta frequência.",
        "disclaimer": "Não é aconselhamento financeiro. As previsões dependem da qualidade dos dados e das hipóteses do modelo.",
        "sim_impact": "Impacto do cenário",
        "sim_prob": "Tabela de probabilidades simuladas",
        "map_metric": "Métrica do mapa",
        "metric_shock": "Probabilidade de choque",
        "metric_growth": "Nowcast médio",
        "question_recession": "Qual é o risco de recessão aqui?",
        "question_drivers": "O que está impulsionando a previsão?",
        "question_models": "Como os modelos estão ponderados?",
    },
    "es": {
        "title": "Simulador ATLAS-GDP",
        "subtitle": "Nowcasts de PIB de frecuencia mixta, pronósticos recursivos multi-horizonte, estrés de escenarios y trazabilidad de auditoría.",
        "sidebar": "Panel institucional de pronóstico del PIB",
        "country": "País",
        "language": "Idioma",
        "as_of": "Fecha de corte",
        "max_h": "Horizonte máximo (trimestres)",
        "scenario": "Simulador de escenarios",
        "tighten": "Ajuste financiero",
        "commodity": "Choque de materias primas",
        "demand": "Impulso de demanda",
        "refresh": "Actualizar artefactos",
        "nowcast": "Nowcast medio",
        "interval50": "Intervalo 50%",
        "interval90": "Intervalo 90%",
        "shock": "Probabilidad de shock",
        "forecast_tab": "Pronóstico",
        "sim_tab": "Simulación",
        "map_tab": "Mapa mundial",
        "backtest_tab": "Backtest",
        "audit_tab": "Auditoría",
        "ask_tab": "Preguntar a ATLAS",
        "about_tab": "Acerca de",
        "blend": "Mezcla de modelos",
        "drivers": "Factores impulsores",
        "driver_chart": "Descomposición de impulsores actuales",
        "backtest": "Resumen del backtest",
        "lineage": "Linaje de datos",
        "forecast_artifact": "Artefacto de pronóstico",
        "drivers_artifact": "Artefacto de impulsores",
        "world_title": "Panorama global actual",
        "top_countries": "Países con mayor riesgo",
        "ask_caption": "Explicador basado en los artefactos actuales.",
        "ask_placeholder": "Haz una pregunta macro sobre el pronóstico actual",
        "about_header": "Qué hace este sistema",
        "about_body": "ATLAS-GDP combina un modelo panel base, un modelo bridge, un modelo factorial dinámico, MIDAS, un BVAR pequeño y un ensemble apilado. Hace nowcast del trimestre actual y extiende esa trayectoria de forma recursiva.",
        "about_methods": "Notas metodológicas",
        "about_methods_body": "El horizonte 0 usa el estado macro actual. Los horizontes siguientes incorporan el pronóstico previo en los rezagos del PIB, mantienen variables estructurales lentas y acercan inflación y desempleo a anclas recientes del país.",
        "about_limits": "Limitaciones",
        "about_limits_body": "El modo offline usa un panel de muestra pequeño. No se modelan revisiones de datos. Las etiquetas trimestrales se simulan a partir de filas anuales cuando faltan publicaciones de alta frecuencia.",
        "disclaimer": "No es asesoría financiera. Los pronósticos dependen de la calidad de los datos y de los supuestos del modelo.",
        "sim_impact": "Impacto del escenario",
        "sim_prob": "Tabla de probabilidades simuladas",
        "map_metric": "Métrica del mapa",
        "metric_shock": "Probabilidad de shock",
        "metric_growth": "Nowcast medio",
        "question_recession": "¿Cuál es el riesgo de recesión aquí?",
        "question_drivers": "¿Qué está impulsando el pronóstico?",
        "question_models": "¿Cómo se ponderan los modelos?",
    },
    "ja": {
        "title": "ATLAS-GDPシミュレーター",
        "subtitle": "混合頻度GDPナウキャスト、再帰的マルチホライズン予測、シナリオ分析、監査向けトレーサビリティ。",
        "sidebar": "機関投資家向けGDP予測ダッシュボード",
        "country": "国",
        "language": "言語",
        "as_of": "基準日",
        "max_h": "最大予測期間（四半期）",
        "scenario": "シナリオシミュレーター",
        "tighten": "金融引き締め",
        "commodity": "商品ショック",
        "demand": "需要押し上げ",
        "refresh": "アーティファクト更新",
        "nowcast": "現在ナウキャスト平均",
        "interval50": "50%区間",
        "interval90": "90%区間",
        "shock": "ショック確率",
        "forecast_tab": "予測",
        "sim_tab": "シミュレーション",
        "map_tab": "世界地図",
        "backtest_tab": "バックテスト",
        "audit_tab": "監査",
        "ask_tab": "ATLASに質問",
        "about_tab": "概要",
        "blend": "モデル配分",
        "drivers": "主要ドライバー",
        "driver_chart": "現在の寄与分解",
        "backtest": "バックテスト概要",
        "lineage": "データ系譜",
        "forecast_artifact": "予測アーティファクト",
        "drivers_artifact": "ドライバーアーティファクト",
        "world_title": "最新グローバルスナップショット",
        "top_countries": "リスク上位の国",
        "ask_caption": "現在のアーティファクトに基づく説明モードです。",
        "ask_placeholder": "現在の予測についてマクロ質問を入力してください",
        "about_header": "このシステムの役割",
        "about_body": "ATLAS-GDPは、ベースラインパネル、ブリッジ、動学因子、MIDAS、小規模BVAR、スタッキング・アンサンブルを組み合わせます。現在四半期をナウキャストし、その先を再帰的に延長します。",
        "about_methods": "手法メモ",
        "about_methods_body": "ホライズン0は現在のマクロ状態を使います。その後のホライズンでは直前の予測をGDPラグに戻し、構造変数は緩やかに維持し、インフレと失業率は直近アンカーへ平均回帰させます。",
        "about_limits": "制約",
        "about_limits_body": "オフラインモードは小さなサンプルパネルを使います。データ改定は扱いません。高頻度系列がない場合、四半期ラベルは年次サンプルから近似されます。",
        "disclaimer": "投資助言ではありません。予測はデータ品質とモデル前提に依存します。",
        "sim_impact": "シナリオ影響",
        "sim_prob": "シミュレーション確率表",
        "map_metric": "地図指標",
        "metric_shock": "ショック確率",
        "metric_growth": "ナウキャスト平均",
        "question_recession": "ここでの景気後退リスクは？",
        "question_drivers": "予測を動かしている要因は？",
        "question_models": "モデルの重みは？",
    },
    "ar": {
        "title": "محاكي ATLAS-GDP",
        "subtitle": "تقديرات فورية للناتج المحلي، وتوقعات متكررة متعددة الآفاق، واختبارات سيناريو، وتتبع قابل للتدقيق.",
        "sidebar": "لوحة مؤسسية لتوقعات الناتج المحلي",
        "country": "الدولة",
        "language": "اللغة",
        "as_of": "تاريخ المرجع",
        "max_h": "أقصى أفق (أرباع)",
        "scenario": "محاكي السيناريو",
        "tighten": "تشديد مالي",
        "commodity": "صدمة سلع",
        "demand": "دعم الطلب",
        "refresh": "تحديث المخرجات",
        "nowcast": "متوسط التقدير الفوري",
        "interval50": "نطاق 50٪",
        "interval90": "نطاق 90٪",
        "shock": "احتمال الصدمة",
        "forecast_tab": "التوقع",
        "sim_tab": "المحاكاة",
        "map_tab": "الخريطة العالمية",
        "backtest_tab": "الاختبار التاريخي",
        "audit_tab": "التدقيق",
        "ask_tab": "اسأل ATLAS",
        "about_tab": "حول",
        "blend": "مزيج النماذج",
        "drivers": "العوامل الدافعة",
        "driver_chart": "تفكيك العوامل الحالية",
        "backtest": "ملخص الاختبار التاريخي",
        "lineage": "سلسلة البيانات",
        "forecast_artifact": "ملف التوقع",
        "drivers_artifact": "ملف العوامل",
        "world_title": "لقطة عالمية حالية",
        "top_countries": "أعلى الدول من حيث المخاطر",
        "ask_caption": "مفسر قائم على المخرجات الحالية.",
        "ask_placeholder": "اطرح سؤالاً اقتصادياً حول التوقع الحالي",
        "about_header": "ماذا يفعل هذا النظام",
        "about_body": "يجمع ATLAS-GDP بين نموذج لوحي أساسي ونموذج bridge ونموذج عامل ديناميكي وMIDAS وBVAR صغير وتجميع تكديسي. ينتج تقديراً فورياً للربع الحالي ثم يمد المسار بشكل متكرر.",
        "about_methods": "ملاحظات المنهجية",
        "about_methods_body": "الأفق صفر يستخدم الحالة الاقتصادية الحالية. الآفاق اللاحقة تعيد إدخال التوقع السابق في تباطؤات الناتج، وتبقي المتغيرات الهيكلية بطيئة الحركة، وتعيد التضخم والبطالة نحو متوسطات حديثة للدولة.",
        "about_limits": "القيود",
        "about_limits_body": "الوضع غير المتصل يستخدم لوحة عينات صغيرة. لا يتم تمثيل مراجعات البيانات. عند غياب الإصدارات عالية التواتر، يتم تقريب الأرباع من صفوف سنوية.",
        "disclaimer": "ليست نصيحة مالية. التوقعات تعتمد على جودة البيانات وافتراضات النموذج.",
        "sim_impact": "أثر السيناريو",
        "sim_prob": "جدول الاحتمالات المحاكاة",
        "map_metric": "مقياس الخريطة",
        "metric_shock": "احتمال الصدمة",
        "metric_growth": "متوسط التقدير الفوري",
        "question_recession": "ما احتمال الركود هنا؟",
        "question_drivers": "ما الذي يدفع التوقع؟",
        "question_models": "كيف يتم وزن النماذج؟",
    },
}


def t(code: str, key: str) -> str:
    return TEXT.get(code, TEXT["en"]).get(key, TEXT["en"].get(key, key))


@st.cache_data(show_spinner=False)
def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def sample_forecast(country: str) -> dict[str, Any]:
    return {
        "country": country,
        "as_of": "2025-02-15",
        "target": "qoq_saar",
        "history": [
            {"period": "2023-Q1", "value": 1.8},
            {"period": "2023-Q2", "value": 1.5},
            {"period": "2023-Q3", "value": 2.0},
            {"period": "2023-Q4", "value": 2.2},
            {"period": "2024-Q1", "value": 2.0},
            {"period": "2024-Q2", "value": 1.6},
            {"period": "2024-Q3", "value": 2.4},
            {"period": "2024-Q4", "value": 2.1},
        ],
        "forecast": [
            {"horizon_q": 0, "period": "2025-Q1", "mean": 2.3, "p10": 1.0, "p25": 1.6, "p50": 2.3, "p75": 2.9, "p90": 3.7},
            {"horizon_q": 1, "period": "2025-Q2", "mean": 2.1, "p10": 0.8, "p25": 1.4, "p50": 2.1, "p75": 2.7, "p90": 3.5},
            {"horizon_q": 2, "period": "2025-Q3", "mean": 2.0, "p10": 0.6, "p25": 1.2, "p50": 2.0, "p75": 2.6, "p90": 3.4},
            {"horizon_q": 4, "period": "2026-Q1", "mean": 1.9, "p10": 0.4, "p25": 1.0, "p50": 1.9, "p75": 2.5, "p90": 3.3},
            {"horizon_q": 8, "period": "2027-Q1", "mean": 1.7, "p10": 0.1, "p25": 0.8, "p50": 1.7, "p75": 2.3, "p90": 3.0},
        ],
        "shock_prob": 0.10,
        "model_weights": {"dfm": 0.30, "midas": 0.20, "bvar": 0.20, "bridge": 0.10, "panel_ml": 0.20},
    }


def sample_world_snapshot() -> pd.DataFrame:
    rows = [
        {"country": "USA", "as_of": "2025-02-15", "mean": 2.3, "p10": 1.0, "p50": 2.3, "p90": 3.7, "shock_prob": 0.10},
        {"country": "CAN", "as_of": "2025-02-15", "mean": 1.7, "p10": 0.3, "p50": 1.7, "p90": 3.1, "shock_prob": 0.18},
        {"country": "GBR", "as_of": "2025-02-15", "mean": 0.9, "p10": -0.3, "p50": 0.9, "p90": 2.0, "shock_prob": 0.28},
        {"country": "NGA", "as_of": "2025-02-15", "mean": 3.1, "p10": 1.2, "p50": 3.1, "p90": 4.8, "shock_prob": 0.14},
        {"country": "IND", "as_of": "2025-02-15", "mean": 6.2, "p10": 4.7, "p50": 6.2, "p90": 7.9, "shock_prob": 0.03},
        {"country": "CHN", "as_of": "2025-02-15", "mean": 4.4, "p10": 3.0, "p50": 4.4, "p90": 5.8, "shock_prob": 0.06},
        {"country": "BRA", "as_of": "2025-02-15", "mean": 2.0, "p10": 0.5, "p50": 2.0, "p90": 3.4, "shock_prob": 0.11},
        {"country": "DEU", "as_of": "2025-02-15", "mean": 0.6, "p10": -0.7, "p50": 0.6, "p90": 1.8, "shock_prob": 0.31},
        {"country": "FRA", "as_of": "2025-02-15", "mean": 1.1, "p10": -0.1, "p50": 1.1, "p90": 2.3, "shock_prob": 0.19},
        {"country": "JPN", "as_of": "2025-02-15", "mean": 0.8, "p10": -0.4, "p50": 0.8, "p90": 1.9, "shock_prob": 0.24},
    ]
    return pd.DataFrame(rows)


def sample_drivers() -> dict[str, Any]:
    return {
        "as_of": "2025-02-15",
        "drivers": [
            {"feature": "gdp_growth_lag1", "contribution": 0.34},
            {"feature": "trade_share", "contribution": 0.18},
            {"feature": "inflation", "contribution": -0.22},
            {"feature": "unemployment", "contribution": -0.19},
            {"feature": "investment_share", "contribution": 0.11},
        ],
        "method": "country_relative_zscore",
    }


def sample_backtest() -> dict[str, Any]:
    return {
        "by_country": {
            "BRA": {"mae": 0.61, "rmse": 0.84, "cov50": 0.54, "cov90": 0.88, "rows": 12},
            "USA": {"mae": 0.49, "rmse": 0.70, "cov50": 0.52, "cov90": 0.86, "rows": 12},
        },
        "overall": {"mae": 0.55, "rmse": 0.77, "cov50": 0.53, "cov90": 0.87, "rows": 24},
        "calibration": [
            {"nominal": 0.5, "observed": 0.53},
            {"nominal": 0.9, "observed": 0.87},
        ],
        "origins": ["2025-Q1", "2025-Q2", "2025-Q3"],
        "horizon_q": 2,
        "interval_method": "residual-derived ensemble quantiles (demo sample)",
    }


def sample_lineage() -> dict[str, Any]:
    return {
        "sources": [
            {"name": "world_bank", "mode": "offline sample", "rows": 140},
            {"name": "forecast_bundle", "path": "artifacts/atlas_gdp_bundle.joblib"},
        ],
        "missingness_rate": 0.04,
        "created_utc": "2026-03-02T20:00:00+00:00",
    }


@st.cache_data(show_spinner=False)
def load_latest_run_manifest() -> dict[str, Any] | None:
    return load_latest_manifest()


def load_forecast_artifacts(manifest: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    forecasts: dict[str, dict[str, Any]] = load_forecast_payloads_from_manifest(manifest) if manifest is not None else {}
    if not forecasts:
        fallback = sample_world_snapshot()
        for country in fallback["country"].tolist():
            forecasts[country] = sample_forecast(country)
    snapshot_path = Path(manifest["artifacts_root"]) / manifest["paths"]["world_snapshot"] if manifest is not None else ART_DIR / "world_snapshot.csv"
    if safe_exists(snapshot_path):
        snapshot = read_csv(snapshot_path)
    else:
        rows = []
        for country, payload in forecasts.items():
            first = pd.DataFrame(payload.get("forecast", [])).iloc[0]
            rows.append(
                {
                    "country": country,
                    "as_of": payload.get("as_of", ""),
                    "mean": float(first.get("mean", 0.0)),
                    "p10": float(first.get("p10", 0.0)),
                    "p50": float(first.get("p50", first.get("mean", 0.0))),
                    "p90": float(first.get("p90", 0.0)),
                    "shock_prob": float(payload.get("shock_prob", 0.0)),
                }
            )
        snapshot = pd.DataFrame(rows)
    return forecasts, snapshot.sort_values("shock_prob", ascending=False)


def load_backtest_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not safe_exists(path):
        return sample_backtest()
    try:
        payload = read_json(path)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    try:
        frame = read_csv(path)
    except Exception:
        return sample_backtest()
    if {"metric", "value"}.issubset(frame.columns):
        pairs = {str(row["metric"]): float(row["value"]) for _, row in frame.iterrows()}
        return {
            "by_country": {},
            "overall": {
                "mae": pairs.get("mae", 0.0),
                "rmse": pairs.get("rmse", 0.0),
                "cov50": pairs.get("coverage_50", 0.0),
                "cov90": pairs.get("coverage_90", pairs.get("coverage_80", 0.0)),
                "rows": int(pairs.get("rows", 0.0)),
            },
            "calibration": [
                {"nominal": 0.5, "observed": pairs.get("coverage_50", 0.0)},
                {"nominal": 0.9, "observed": pairs.get("coverage_90", pairs.get("coverage_80", 0.0))},
            ],
            "origins": [],
            "horizon_q": 1,
            "interval_method": "legacy metric CSV",
        }
    return sample_backtest()


def apply_scenario(fcst: pd.DataFrame, tighten: float, commodity: float, demand: float) -> pd.DataFrame:
    out = fcst.copy()
    shift = (-0.25 * tighten) + (-0.15 * commodity) + (0.20 * demand)
    dispersion = abs(tighten) * 0.08 + abs(commodity) * 0.06 + abs(demand) * 0.04
    for col in ["mean", "p10", "p25", "p50", "p75", "p90"]:
        out[col] = out[col] + shift
    out["p10"] = out["p10"] - dispersion
    out["p25"] = out["p25"] - dispersion / 2.0
    out["p75"] = out["p75"] + dispersion / 2.0
    out["p90"] = out["p90"] + dispersion
    return out


def fan_chart(history: pd.DataFrame, fcst: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["period"], y=history["value"], mode="lines+markers", name="History", line=dict(color="#1f2937")))
    fig.add_trace(go.Scatter(x=fcst["period"], y=fcst["p90"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fcst["period"], y=fcst["p10"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(59,130,246,0.15)", name="90%"))
    fig.add_trace(go.Scatter(x=fcst["period"], y=fcst["p75"], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=fcst["period"], y=fcst["p25"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(37,99,235,0.28)", name="50%"))
    fig.add_trace(go.Scatter(x=fcst["period"], y=fcst["p50"], mode="lines+markers", name="Median", line=dict(color="#2563eb", width=3)))
    fig.update_layout(title=title, xaxis_title="Period", yaxis_title="GDP growth", hovermode="x unified", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def drivers_bar(drivers_df: pd.DataFrame, title: str) -> go.Figure:
    ordered = drivers_df.sort_values("contribution")
    fig = go.Figure(go.Bar(x=ordered["contribution"], y=ordered["feature"], orientation="h", marker_color=["#dc2626" if x < 0 else "#16a34a" for x in ordered["contribution"]]))
    fig.update_layout(title=title, xaxis_title="Contribution", yaxis_title="Feature", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def comparison_chart(trace_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col in ["ridge", "hgb", "bridge", "dfm", "midas", "bvar", "mean"]:
        if col in trace_df.columns:
            name = "ensemble" if col == "mean" else col
            fig.add_trace(go.Scatter(x=trace_df["horizon_q"], y=trace_df[col], mode="lines+markers", name=name))
    fig.update_layout(title=title, xaxis_title="Horizon (quarters)", yaxis_title="Forecast", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def simulation_table(fcst: pd.DataFrame) -> pd.DataFrame:
    first = fcst.iloc[0]
    width = max(0.25, float(first["p90"] - first["p10"]))
    std = width / 2.56
    draws = np.random.default_rng(42).normal(float(first["mean"]), std, size=5000)
    return pd.DataFrame(
        [
            {"scenario": "GDP below 0%", "probability": float(np.mean(draws < 0.0))},
            {"scenario": "GDP below 1%", "probability": float(np.mean(draws < 1.0))},
            {"scenario": "GDP above 2.5%", "probability": float(np.mean(draws > 2.5))},
            {"scenario": "GDP above 3.5%", "probability": float(np.mean(draws > 3.5))},
        ]
    )


def apply_snapshot_scenario(snapshot: pd.DataFrame, tighten: float, commodity: float, demand: float) -> pd.DataFrame:
    out = snapshot.copy()
    shift = (-0.25 * tighten) + (-0.15 * commodity) + (0.20 * demand)
    out["mean"] = out["mean"] + shift
    out["p50"] = out.get("p50", out["mean"]) + shift
    out["shock_prob"] = np.clip(out["shock_prob"] + 0.06 * np.maximum(tighten, 0) + 0.04 * np.maximum(commodity, 0) - 0.03 * np.maximum(demand, 0), 0.0, 1.0)
    return out


def world_map(snapshot: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = go.Figure(
        go.Choropleth(
            locations=snapshot["country"],
            z=snapshot[metric],
            locationmode="ISO-3",
            colorscale="Blues" if metric != "shock_prob" else "Reds",
            colorbar_title=metric,
            marker_line_color="white",
        )
    )
    fig.update_layout(title=title, geo=dict(showframe=False, showcoastlines=True, projection_type="equirectangular"), margin=dict(l=10, r=10, t=40, b=10))
    return fig


def make_chat_answer(code: str, question: str, country: str, forecast_df: pd.DataFrame, drivers_df: pd.DataFrame, weights: dict[str, float], scenario_shift: float) -> str:
    lower = question.lower()
    now = forecast_df.iloc[0]
    top_pos = drivers_df.sort_values("contribution", ascending=False).head(2)
    top_neg = drivers_df.sort_values("contribution").head(2)
    pos = ", ".join(f"{row.feature} ({row.contribution:+.2f})" for row in top_pos.itertuples()) if not top_pos.empty else "n/a"
    neg = ", ".join(f"{row.feature} ({row.contribution:+.2f})" for row in top_neg.itertuples()) if not top_neg.empty else "n/a"
    if "recession" in lower or "ركود" in lower or "récession" in lower:
        return {
            "en": f"For {country}, the simulated probability of sub-zero growth is materially linked to the left tail. The current median is {now['p50']:.2f} and the 90% band is [{now['p10']:.2f}, {now['p90']:.2f}]. Scenario shift applied: {scenario_shift:+.2f}.",
            "fr": f"Pour {country}, le risque de croissance négative vient surtout de la queue gauche. La médiane actuelle est {now['p50']:.2f} et la bande à 90 % est [{now['p10']:.2f}, {now['p90']:.2f}]. Décalage de scénario: {scenario_shift:+.2f}.",
            "pt": f"Para {country}, o risco de crescimento negativo está ligado à cauda esquerda. A mediana atual é {now['p50']:.2f} e a banda de 90% é [{now['p10']:.2f}, {now['p90']:.2f}]. Deslocamento do cenário: {scenario_shift:+.2f}.",
            "es": f"Para {country}, el riesgo de crecimiento negativo viene de la cola izquierda. La mediana actual es {now['p50']:.2f} y la banda del 90% es [{now['p10']:.2f}, {now['p90']:.2f}]. Desplazamiento del escenario: {scenario_shift:+.2f}.",
            "ja": f"{country}では、マイナス成長リスクは左側テールに集中しています。現在の中央値は{now['p50']:.2f}、90%区間は[{now['p10']:.2f}, {now['p90']:.2f}]です。シナリオ調整は{scenario_shift:+.2f}です。",
            "ar": f"بالنسبة إلى {country} فإن خطر النمو السلبي يرتبط بالطرف الأيسر للتوزيع. الوسيط الحالي هو {now['p50']:.2f} ونطاق 90٪ هو [{now['p10']:.2f}, {now['p90']:.2f}]. تأثير السيناريو هو {scenario_shift:+.2f}.",
        }[code]
    if "driver" in lower or "facteur" in lower or "impul" in lower or "要因" in lower or "العوامل" in lower:
        return {
            "en": f"The forecast is lifted by {pos} and restrained by {neg}.",
            "fr": f"La prévision est soutenue par {pos} et freinée par {neg}.",
            "pt": f"A previsão é puxada por {pos} e contida por {neg}.",
            "es": f"El pronóstico sube por {pos} y se frena por {neg}.",
            "ja": f"予測を押し上げているのは {pos}、下押ししているのは {neg} です。",
            "ar": f"التوقع مدعوم بواسطة {pos} ويُقيد بواسطة {neg}.",
        }[code]
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    weight_text = ", ".join(f"{k}: {v:.2f}" for k, v in ordered)
    return {
        "en": f"ATLAS is using the current saved artifact state for {country}. The model blend is {weight_text}.",
        "fr": f"ATLAS utilise l’état d’artefact actuel pour {country}. La pondération est {weight_text}.",
        "pt": f"O ATLAS usa o estado atual dos artefatos para {country}. A mistura de modelos é {weight_text}.",
        "es": f"ATLAS usa el estado actual de los artefactos para {country}. La mezcla de modelos es {weight_text}.",
        "ja": f"ATLASは{country}の保存済みアーティファクトを使っています。モデル配分は {weight_text} です。",
        "ar": f"يستخدم ATLAS حالة المخرجات الحالية لـ {country}. مزيج النماذج هو {weight_text}.",
    }[code]


language_name = st.sidebar.selectbox("Language / Langue / Idioma", LANGUAGES)
lang = LANG_CODES[language_name]
rtl = lang == "ar"
if rtl:
    st.markdown('<div dir="rtl">', unsafe_allow_html=True)

settings = load_settings()
ART_DIR = settings.paths.artifacts / "latest"
st.sidebar.title("📈 " + t(lang, "title"))
st.sidebar.caption(t(lang, "sidebar"))
as_of_value = st.sidebar.date_input(t(lang, "as_of"), value=date.today())
max_h = st.sidebar.slider(t(lang, "max_h"), min_value=0, max_value=8, value=4)

st.sidebar.subheader(t(lang, "scenario"))
tighten = st.sidebar.slider(t(lang, "tighten"), -2.0, 2.0, 0.0, 0.1)
commodity = st.sidebar.slider(t(lang, "commodity"), -2.0, 2.0, 0.0, 0.1)
demand = st.sidebar.slider(t(lang, "demand"), -2.0, 2.0, 0.0, 0.1)
st.sidebar.caption(t(lang, "refresh"))
manifest = load_latest_run_manifest()
manifest_countries = manifest_country_list(manifest) if manifest is not None else list(settings.raw_config.get("data", {}).get("countries", []))
if not manifest_countries:
    manifest_countries = ["USA"]
country = st.sidebar.selectbox(t(lang, "country"), manifest_countries)
refresh_clicked = st.sidebar.button("Run / Refresh (recompute)")
if manifest is None:
    with st.status("Running initial artifact build...", expanded=True) as status:
        status.write("Preparing baseline run for the configured countries.")
        run_result = run_pipeline(
            as_of=as_of_value,
            countries=manifest_countries,
            horizon_q=max_h,
            scenario={
                "financial_tightening": 0.0,
                "commodity_shock": 0.0,
                "demand_boost": 0.0,
            },
            offline=settings.offline_mode,
        )
        st.session_state["atlas_last_run_id"] = run_result.run_id
        status.update(label="Initial run complete", state="complete")
    st.cache_data.clear()
    manifest = load_latest_run_manifest()
elif refresh_clicked:
    progress = st.progress(0, text="Starting recompute")
    with st.status("Running pipeline recompute...", expanded=True) as status:
        status.write(f"Building artifacts for {country} as of {as_of_value.isoformat()}.")
        progress.progress(25, text="Launching pipeline")
        run_result = run_pipeline(
            as_of=as_of_value,
            countries=[country],
            horizon_q=max_h,
            scenario={
                "financial_tightening": tighten,
                "commodity_shock": commodity,
                "demand_boost": demand,
            },
            offline=settings.offline_mode,
        )
        progress.progress(90, text="Loading manifest")
        st.session_state["atlas_last_run_id"] = run_result.run_id
        st.session_state["atlas_last_run_at"] = run_result.created_at_utc
        status.write(f"Run completed: {run_result.run_id}")
        status.update(label="Recompute complete", state="complete")
    progress.progress(100, text="Done")
    st.cache_data.clear()
    manifest = load_latest_run_manifest()

forecasts, snapshot = load_forecast_artifacts(manifest)
countries = sorted(forecasts.keys())
if country not in countries:
    country = countries[0]

forecast_payload = forecasts.get(country, sample_forecast(country))
drivers_path = Path(manifest["artifacts_root"]) / manifest["paths"]["drivers"][country] if manifest and country in manifest.get("paths", {}).get("drivers", {}) else ART_DIR / f"drivers_{country}.json"
if safe_exists(drivers_path):
    drivers_payload = read_json(drivers_path)
else:
    drivers_payload = sample_drivers()
metrics_name = manifest.get("paths", {}).get("backtest_report") if manifest else None
metrics_path = Path(manifest["artifacts_root"]) / metrics_name if manifest and metrics_name else None
backtest_summary = load_backtest_summary(metrics_path)
lineage_path = Path(manifest["artifacts_root"]) / "data_lineage.json" if manifest else ART_DIR / "data_lineage.json"
lineage = read_json(lineage_path) if safe_exists(lineage_path) else sample_lineage()

history_df = pd.DataFrame(forecast_payload.get("history", []))
baseline_forecast_df = pd.DataFrame(forecast_payload.get("forecast", []))
baseline_forecast_df = baseline_forecast_df[baseline_forecast_df["horizon_q"] <= max_h].copy() if not baseline_forecast_df.empty else pd.DataFrame(sample_forecast(country)["forecast"])
forecast_df = apply_scenario(baseline_forecast_df, tighten, commodity, demand)
driver_df = pd.DataFrame(drivers_payload.get("drivers", []))
trace_df = pd.DataFrame(forecast_payload.get("trace", []))
weights = forecast_payload.get("model_weights", {})
scenario_shift = (-0.25 * tighten) + (-0.15 * commodity) + (0.20 * demand)
shock_prob = float(np.clip(float(forecast_payload.get("shock_prob", 0.0)) + 0.06 * max(tighten, 0) + 0.04 * max(commodity, 0) - 0.03 * max(demand, 0), 0.0, 1.0))
snapshot = apply_snapshot_scenario(snapshot, tighten, commodity, demand)
run_id = manifest.get("run_id", "sample-run") if manifest else "sample-run"
run_created = manifest.get("created_at_utc", "n/a") if manifest else "n/a"

st.title(t(lang, "title"))
st.caption(t(lang, "subtitle"))
st.caption(f"Last run: `{run_id}` at `{run_created}`")
if manifest is not None and manifest.get("as_of") != as_of_value.isoformat():
    st.warning(f"Artifact viewer mode: current run is for {manifest.get('as_of')}. Click Refresh to recompute for {as_of_value.isoformat()}.")

now = forecast_df.iloc[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric(t(lang, "nowcast"), f"{now['mean']:.2f}")
k2.metric(t(lang, "interval50"), f"[{now['p25']:.2f}, {now['p75']:.2f}]")
k3.metric(t(lang, "interval90"), f"[{now['p10']:.2f}, {now['p90']:.2f}]")
k4.metric(t(lang, "shock"), f"{100.0 * shock_prob:.1f}%")

tabs = st.tabs([
    t(lang, "forecast_tab"),
    t(lang, "sim_tab"),
    t(lang, "map_tab"),
    t(lang, "backtest_tab"),
    t(lang, "audit_tab"),
    t(lang, "ask_tab"),
    t(lang, "about_tab"),
])

with tabs[0]:
    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(fan_chart(history_df, forecast_df, f"{country} — {t(lang, 'forecast_tab')}"), width="stretch")
        if not trace_df.empty:
            st.plotly_chart(comparison_chart(trace_df, f"{country} model path comparison"), width="stretch")
    with right:
        st.subheader(t(lang, "blend"))
        if weights:
            weights_df = pd.DataFrame({"model": list(weights.keys()), "weight": list(weights.values())}).sort_values("weight", ascending=False)
            st.dataframe(weights_df, width="stretch", hide_index=True)
        st.subheader(t(lang, "drivers"))
        if not driver_df.empty:
            st.plotly_chart(drivers_bar(driver_df, t(lang, "driver_chart")), width="stretch")
        else:
            st.info("No drivers available.")

with tabs[1]:
    st.subheader(t(lang, "sim_impact"))
    st.caption("Scenario sliders are applied client-side on top of the selected baseline run.")
    impact_df = pd.DataFrame([
        {"lever": t(lang, "tighten"), "input": tighten, "forecast_shift": -0.25 * tighten},
        {"lever": t(lang, "commodity"), "input": commodity, "forecast_shift": -0.15 * commodity},
        {"lever": t(lang, "demand"), "input": demand, "forecast_shift": 0.20 * demand},
    ])
    scenario_delta = forecast_df[["horizon_q", "period", "mean"]].merge(
        baseline_forecast_df[["horizon_q", "mean"]],
        on="horizon_q",
        how="left",
        suffixes=("", "_baseline"),
    )
    scenario_delta["delta_vs_baseline"] = scenario_delta["mean"] - scenario_delta["mean_baseline"]
    left, right = st.columns(2)
    with left:
        st.dataframe(impact_df, width="stretch", hide_index=True)
        st.dataframe(scenario_delta[["period", "delta_vs_baseline"]], width="stretch", hide_index=True)
        st.download_button(
            "Download scenario JSON",
            data=json.dumps(
                {
                    "run_id": run_id,
                    "country": country,
                    "as_of": as_of_value.isoformat(),
                    "scenario": {"tighten": tighten, "commodity": commodity, "demand": demand},
                    "delta": scenario_delta[["period", "delta_vs_baseline"]].to_dict(orient="records"),
                },
                indent=2,
            ),
            file_name=f"atlas_scenario_{country}_{run_id}.json",
            mime="application/json",
        )
    with right:
        sim_df = simulation_table(forecast_df)
        sim_df["probability"] = (100.0 * sim_df["probability"]).round(1).astype(str) + "%"
        st.subheader(t(lang, "sim_prob"))
        st.dataframe(sim_df, width="stretch", hide_index=True)
    st.plotly_chart(fan_chart(history_df, forecast_df, f"{country} stressed forecast"), width="stretch")

with tabs[2]:
    map_metric = st.selectbox(t(lang, "map_metric"), [t(lang, "metric_shock"), t(lang, "metric_growth")])
    metric_key = "shock_prob" if map_metric == t(lang, "metric_shock") else "mean"
    st.plotly_chart(world_map(snapshot, metric_key, t(lang, "world_title")), width="stretch")
    st.subheader(t(lang, "top_countries"))
    top_cols = ["country", metric_key, "p10", "p90"] if metric_key == "mean" else ["country", metric_key, "mean", "p10", "p90"]
    st.dataframe(snapshot.sort_values(metric_key, ascending=False).head(10)[top_cols], width="stretch", hide_index=True)

with tabs[3]:
    st.subheader(t(lang, "backtest"))
    st.caption("Rolling-origin evaluation from the latest pipeline run.")
    by_country_df = pd.DataFrame(
        [{"country": key, **value} for key, value in backtest_summary.get("by_country", {}).items()]
    )
    if not by_country_df.empty:
        st.dataframe(by_country_df, width="stretch", hide_index=True)
    overall = backtest_summary.get("overall", {})
    if overall:
        st.caption(
            f"Overall MAE {float(overall.get('mae', 0.0)):.2f} | RMSE {float(overall.get('rmse', 0.0)):.2f} | "
            f"50% cov {100.0 * float(overall.get('cov50', 0.0)):.1f}% | 90% cov {100.0 * float(overall.get('cov90', 0.0)):.1f}%"
        )
    calibration_df = pd.DataFrame(backtest_summary.get("calibration", []))
    if not calibration_df.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line=dict(color="#94a3b8", dash="dash")))
        fig.add_trace(
            go.Scatter(
                x=calibration_df["nominal"],
                y=calibration_df["observed"],
                mode="lines+markers",
                name="Observed",
                line=dict(color="#2563eb"),
            )
        )
        fig.update_layout(
            title="Coverage vs nominal",
            xaxis_title="Nominal coverage",
            yaxis_title="Observed coverage",
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width="stretch")
    backtest_plot_name = manifest.get("paths", {}).get("backtest_plot") if manifest else None
    backtest_plot_path = Path(manifest["artifacts_root"]) / backtest_plot_name if manifest and backtest_plot_name else None
    if backtest_plot_path is not None and safe_exists(backtest_plot_path):
        st.image(str(backtest_plot_path), caption="Saved coverage calibration plot")

with tabs[4]:
    st.subheader(t(lang, "lineage"))
    st.json(lineage)
    st.subheader("Connector Status")
    st.json((manifest or {}).get("connector_status", lineage.get("connector_status", {})))
    st.subheader("Settings Snapshot")
    st.json((manifest or {}).get("settings_snapshot", {}))
    st.subheader("Config Snapshot")
    st.json((manifest or {}).get("config_snapshot", {}))
    st.subheader("Run Manifest")
    st.json(manifest or {"mode": "sample"})
    st.subheader(t(lang, "forecast_artifact"))
    st.json(forecast_payload)
    st.subheader(t(lang, "drivers_artifact"))
    st.json(drivers_payload)

with tabs[5]:
    st.subheader(t(lang, "ask_tab"))
    if not settings.ask_atlas_enabled or not settings.ask_atlas_api_key:
        st.info("Ask ATLAS is disabled. Set ASK_ATLAS_ENABLED=1 and ASK_ATLAS_API_KEY to enable it.")
    else:
        st.caption(t(lang, "ask_caption"))
        for preset in [t(lang, "question_recession"), t(lang, "question_drivers"), t(lang, "question_models")]:
            if st.button(preset, key=f"preset_{preset}"):
                st.session_state["atlas_answer"] = make_chat_answer(lang, preset, country, forecast_df, driver_df, weights, scenario_shift)
                st.session_state["atlas_prompt"] = preset
        question = st.chat_input(t(lang, "ask_placeholder"))
        if question:
            st.session_state["atlas_answer"] = make_chat_answer(lang, question, country, forecast_df, driver_df, weights, scenario_shift)
            st.session_state["atlas_prompt"] = question
        if "atlas_answer" in st.session_state:
            prompt = st.session_state.get("atlas_prompt", "")
            if prompt:
                st.markdown(f"**Q:** {prompt}")
            st.success(st.session_state["atlas_answer"])

with tabs[6]:
    st.subheader(t(lang, "about_header"))
    st.write("ATLAS-GDP is a mixed-frequency nowcasting demo with quarterly-aligned features, manifest-based artifacts, and rolling-origin evaluation.")
    st.subheader(t(lang, "about_methods"))
    st.write("Refresh runs the local pipeline, writes a new run folder and manifest, and updates the latest run pointer. Scenario sliders apply a transparent stress overlay on the selected baseline run.")
    st.subheader(t(lang, "about_limits"))
    st.write("This remains a demo system. OECD, IMF, and BEA enrichments are demo-only unless local files are provided. The monthly-to-quarter aggregation is simplified and not a full vintage data engine.")

st.caption(t(lang, "disclaimer"))
if rtl:
    st.markdown('</div>', unsafe_allow_html=True)
