import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="GPC Batch Analyzer", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Zen+Maru+Gothic&display=swap');

html, body, [class*="css"] {
    font-family: 'Zen Maru Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

@dataclass
class AnalysisResult:
    sheet_name: str
    sample_name: str
    baseline_x: float
    baseline_y_raw: float
    point2_x: float
    point2_y_raw: float
    point3_x: float
    point3_y_raw: float
    point4_x: float
    point4_y_final: float
    area1: float
    area2: float
    area3: float
    total_area: float
    area1_abs: float
    area2_abs: float
    area3_abs: float
    ratio1: float
    ratio2: float
    ratio3: float
    m1: float
    b1: float
    m2: float
    b2: float


def extract_sample_name(df: pd.DataFrame) -> str:
    """Extract sample name from the 'Data File Name' row."""
    try:
        for i in range(len(df)):
            if str(df.iloc[i, 0]).strip() == "Data File Name":
                full_path = str(df.iloc[i, 1]).strip()
                # Support both Windows and Unix separators
                file_name = re.split(r"[\\/]", full_path)[-1]
                file_name = re.sub(r"\.lcd$", "", file_name, flags=re.IGNORECASE)
                return file_name
    except Exception:
        pass
    return "Unknown sample"


def load_sheet_data(df: pd.DataFrame) -> pd.DataFrame:
    """Use row 25 onward (0-based index 24) and first two columns as x/y."""
    data = df.iloc[24:, :2].copy()
    data.columns = ["x", "y"]
    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["x", "y"]).sort_values("x").reset_index(drop=True)
    if data.empty:
        raise ValueError("25行目以降に有効な数値データがありません。")
    return data


def get_min_point(df: pd.DataFrame, x1: float, x2: float, y_col: str = "y") -> Tuple[float, float]:
    start, end = sorted([x1, x2])
    sub = df[(df["x"] >= start) & (df["x"] <= end)].copy()
    if sub.empty:
        raise ValueError(f"指定範囲 {start}–{end} にデータがありません。")
    row = sub.loc[sub[y_col].idxmin()]
    return float(row["x"]), float(row[y_col])


def analyze_sheet(
    df_raw: pd.DataFrame,
    sheet_name: str,
    x_view_min: float,
    x_view_max: float,
    baseline_range: Tuple[float, float],
    point2_range: Tuple[float, float],
    point3_range: Tuple[float, float],
    point4_range: Tuple[float, float],
) -> Tuple[AnalysisResult, Dict[str, pd.DataFrame]]:
    sample_name = extract_sample_name(df_raw)
    plot_df = load_sheet_data(df_raw)

    # baseline, point2, point3 are found on raw y values
    baseline_x, baseline_min = get_min_point(plot_df, baseline_range[0], baseline_range[1], "y")
    point2_x, point2_y_raw = get_min_point(plot_df, point2_range[0], point2_range[1], "y")
    point3_x, point3_y_raw = get_min_point(plot_df, point3_range[0], point3_range[1], "y")

    baseline_y_corrected = 0.0
    point2_y_corrected = point2_y_raw - baseline_min
    point3_y_corrected = point3_y_raw - baseline_min

    if not (baseline_x <= point2_x <= point3_x):
        raise ValueError(
            f"x座標の順序が baseline <= point2 <= point3 になっていません: "
            f"{baseline_x:.3f}, {point2_x:.3f}, {point3_x:.3f}"
        )

    target_df = plot_df[(plot_df["x"] >= baseline_x) & (plot_df["x"] <= point3_x)].copy()
    if target_df.empty:
        raise ValueError("baseline から point3 の範囲にデータがありません。")

    target_df["y_corrected_base"] = target_df["y"] - baseline_min

    # piecewise lines
    if point2_x == baseline_x:
        raise ValueError("baseline_x と point2_x が同じため、直線1を計算できません。")
    if point3_x == point2_x:
        raise ValueError("point2_x と point3_x が同じため、直線2を計算できません。")

    m1 = (point2_y_corrected - baseline_y_corrected) / (point2_x - baseline_x)
    b1 = baseline_y_corrected - m1 * baseline_x
    m2 = (point3_y_corrected - point2_y_corrected) / (point3_x - point2_x)
    b2 = point2_y_corrected - m2 * point2_x

    target_df["line_y"] = np.nan
    mask1 = (target_df["x"] >= baseline_x) & (target_df["x"] <= point2_x)
    mask2 = (target_df["x"] > point2_x) & (target_df["x"] <= point3_x)
    target_df.loc[mask1, "line_y"] = m1 * target_df.loc[mask1, "x"] + b1
    target_df.loc[mask2, "line_y"] = m2 * target_df.loc[mask2, "x"] + b2
    target_df["y_final"] = target_df["y_corrected_base"] - target_df["line_y"]

    # point4 is found on final corrected data
    point4_x, point4_y_final = get_min_point(target_df, point4_range[0], point4_range[1], "y_final")

    if not (baseline_x <= point2_x <= point4_x <= point3_x):
        raise ValueError(
            f"x座標の順序が baseline <= point2 <= point4 <= point3 になっていません: "
            f"{baseline_x:.3f}, {point2_x:.3f}, {point4_x:.3f}, {point3_x:.3f}"
        )

    area1_df = target_df[(target_df["x"] >= baseline_x) & (target_df["x"] <= point2_x)].copy()
    area2_df = target_df[(target_df["x"] >= point2_x) & (target_df["x"] <= point4_x)].copy()
    area3_df = target_df[(target_df["x"] >= point4_x) & (target_df["x"] <= point3_x)].copy()

    if area1_df.empty or area2_df.empty or area3_df.empty:
        raise ValueError("いずれかの面積計算範囲にデータがありません。")

    area1 = float(np.trapezoid(area1_df["y_final"], area1_df["x"]))
    area2 = float(np.trapezoid(area2_df["y_final"], area2_df["x"]))
    area3 = float(np.trapezoid(area3_df["y_final"], area3_df["x"]))
    total_area = area1 + area2 + area3

    area1_abs = float(np.trapezoid(np.abs(area1_df["y_final"]), area1_df["x"]))
    area2_abs = float(np.trapezoid(np.abs(area2_df["y_final"]), area2_df["x"]))
    area3_abs = float(np.trapezoid(np.abs(area3_df["y_final"]), area3_df["x"]))

    if total_area != 0:
        ratio1 = area1 / total_area * 100.0
        ratio2 = area2 / total_area * 100.0
        ratio3 = area3 / total_area * 100.0
    else:
        ratio1 = ratio2 = ratio3 = np.nan

    result = AnalysisResult(
        sheet_name=sheet_name,
        sample_name=sample_name,
        baseline_x=float(baseline_x),
        baseline_y_raw=float(baseline_min),
        point2_x=float(point2_x),
        point2_y_raw=float(point2_y_raw),
        point3_x=float(point3_x),
        point3_y_raw=float(point3_y_raw),
        point4_x=float(point4_x),
        point4_y_final=float(point4_y_final),
        area1=area1,
        area2=area2,
        area3=area3,
        total_area=total_area,
        area1_abs=area1_abs,
        area2_abs=area2_abs,
        area3_abs=area3_abs,
        ratio1=ratio1,
        ratio2=ratio2,
        ratio3=ratio3,
        m1=float(m1),
        b1=float(b1),
        m2=float(m2),
        b2=float(b2),
    )

    context = {
        "plot_df": plot_df,
        "zoom_df": plot_df[(plot_df["x"] >= min(x_view_min, x_view_max)) & (plot_df["x"] <= max(x_view_min, x_view_max))].copy(),
        "target_df": target_df,
        "area1_df": area1_df,
        "area2_df": area2_df,
        "area3_df": area3_df,
    }
    return result, context


def make_overview_plot(plot_df: pd.DataFrame, sample_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["x"], y=plot_df["y"],
        mode="lines+markers",
        name="raw data",
        hovertemplate="x=%{x}<br>y=%{y}<extra></extra>"
    ))
    fig.update_layout(title=f"Raw Plot: {sample_name}", xaxis_title="x", yaxis_title="y")
    return fig


def make_zoom_plot(zoom_df: pd.DataFrame, sample_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=zoom_df["x"], y=zoom_df["y"],
        mode="lines+markers",
        name="zoomed data",
        hovertemplate="x=%{x}<br>y=%{y}<extra></extra>"
    ))
    fig.update_layout(title=f"Zoomed Plot: {sample_name}", xaxis_title="x", yaxis_title="y")
    return fig


def make_final_plot(result: AnalysisResult, ctx: Dict[str, pd.DataFrame]) -> go.Figure:
    target_df = ctx["target_df"]
    area1_df = ctx["area1_df"]
    area2_df = ctx["area2_df"]
    area3_df = ctx["area3_df"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=target_df["x"], y=target_df["y_final"],
        mode="lines+markers",
        name="corrected data",
        hovertemplate="x=%{x}<br>y=%{y}<extra></extra>"
    ))

    for area_df, name in [
        (area1_df, f"Area 1 = {result.area1:.3f}"),
        (area2_df, f"Area 2 = {result.area2:.3f}"),
        (area3_df, f"Area 3 = {result.area3:.3f}"),
    ]:
        fig.add_trace(go.Scatter(
            x=area_df["x"], y=area_df["y_final"],
            mode="lines", fill="tozeroy", name=name,
            hovertemplate="x=%{x}<br>y=%{y}<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=[result.baseline_x, result.point2_x, result.point3_x],
        y=[0, 0, 0],
        mode="markers",
        marker=dict(size=12, symbol="circle-open", line=dict(width=2)),
        name="baseline / point2 / point3",
        hovertemplate="x=%{x}<br>y=%{y}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[result.point4_x], y=[result.point4_y_final],
        mode="markers",
        marker=dict(size=12, symbol="circle-open", line=dict(width=2)),
        name="point4",
        hovertemplate="x=%{x}<br>y=%{y}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Final Corrected Plot with Areas: {result.sample_name}",
        xaxis_title="x",
        yaxis_title="y_final",
        xaxis=dict(range=[result.baseline_x, result.point3_x]),
    )
    return fig


def results_to_dataframe(results: List[AnalysisResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def to_excel_bytes(results_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="results")
    return output.getvalue()


st.title("GFCTool")
st.write(
    "ゲル濾過クロマトグラフィーの波形データが入ったExcelファイル内の全シートを順番に処理し、"
    "アミロース/短鎖アミロペクチン/長鎖アミロペクチンの測定結果を出力します。"
)

with st.sidebar:
    st.header("解析範囲")
    x_view_min = st.number_input("左端", value=80.0)
    x_view_max = st.number_input("右端", value=190.0)

    st.subheader("ベースライン決定に使う範囲")
    baseline_x1 = st.number_input("左端", value=87.0)
    baseline_x2 = st.number_input("右端", value=90.0)

    st.subheader("point2 範囲")
    point2_x1 = st.number_input("point2_x1", value=100.0)
    point2_x2 = st.number_input("point2_x2", value=120.0)

    st.subheader("point3 範囲")
    point3_x1 = st.number_input("point3_x1", value=160.0)
    point3_x2 = st.number_input("point3_x2", value=180.0)

    st.subheader("point4 範囲")
    point4_x1 = st.number_input("point4_x1", value=130.0)
    point4_x2 = st.number_input("point4_x2", value=140.0)
    
st.info("左または上のアップロード欄から Excel ファイルを選んでください。")
uploaded_file = st.file_uploader("", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        xls = pd.ExcelFile(uploaded_file)
        st.success(f"{len(xls.sheet_names)} 個のシートを検出しました。")

        if st.button("全サンプルを解析", type="primary"):
            all_results: List[AnalysisResult] = []
            all_contexts: Dict[str, Dict[str, pd.DataFrame]] = {}
            errors: List[Dict[str, str]] = []

            progress = st.progress(0)
            status = st.empty()

            for i, sheet_name in enumerate(xls.sheet_names):
                status.write(f"解析中: {sheet_name}")
                try:
                    df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None)
                    result, ctx = analyze_sheet(
                        df_raw=df_raw,
                        sheet_name=sheet_name,
                        x_view_min=x_view_min,
                        x_view_max=x_view_max,
                        baseline_range=(baseline_x1, baseline_x2),
                        point2_range=(point2_x1, point2_x2),
                        point3_range=(point3_x1, point3_x2),
                        point4_range=(point4_x1, point4_x2),
                    )
                    all_results.append(result)
                    all_contexts[sheet_name] = ctx
                except Exception as e:
                    errors.append({"sheet_name": sheet_name, "error": str(e)})
                progress.progress((i + 1) / len(xls.sheet_names))

            status.write("解析完了")
            st.session_state["all_results"] = all_results
            st.session_state["all_contexts"] = all_contexts
            st.session_state["errors"] = errors

        if "all_results" in st.session_state and st.session_state["all_results"]:
            results: List[AnalysisResult] = st.session_state["all_results"]
            contexts: Dict[str, Dict[str, pd.DataFrame]] = st.session_state["all_contexts"]
            errors = st.session_state.get("errors", [])

            results_df = results_to_dataframe(results)

            st.subheader("解析結果一覧")
            st.dataframe(results_df, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "結果をCSVでダウンロード",
                    data=results_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="gpc_analysis_results.csv",
                    mime="text/csv",
                )
            with c2:
                st.download_button(
                    "結果をExcelでダウンロード",
                    data=to_excel_bytes(results_df),
                    file_name="gpc_analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            if errors:
                st.subheader("エラーが出たシート")
                st.dataframe(pd.DataFrame(errors), use_container_width=True)

            st.subheader("個別サンプルの詳細表示")
            sample_options = [f"{r.sheet_name} | {r.sample_name}" for r in results]
            selected_label = st.selectbox("表示するサンプルを選択", sample_options)
            selected_sheet = selected_label.split(" | ")[0]
            selected_result = next(r for r in results if r.sheet_name == selected_sheet)
            selected_ctx = contexts[selected_sheet]

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("アミロース", f"{selected_result.area1:.4f}")
            mcol2.metric("長鎖アミロペクチン", f"{selected_result.area2:.4f}")
            mcol3.metric("短鎖アミロペクチン", f"{selected_result.area3:.4f}")
            #mcol4.metric("Total Area", f"{selected_result.total_area:.4f}")

            scol1, scol2 = st.columns(2)
            with scol1:
                st.plotly_chart(make_overview_plot(selected_ctx["plot_df"], selected_result.sample_name), use_container_width=True)
            with scol2:
                st.plotly_chart(make_zoom_plot(selected_ctx["zoom_df"], selected_result.sample_name), use_container_width=True)

            st.plotly_chart(make_final_plot(selected_result, selected_ctx), use_container_width=True)

            detail_df = pd.DataFrame([
                {"item": "sheet_name", "value": selected_result.sheet_name},
                {"item": "sample_name", "value": selected_result.sample_name},
                {"item": "baseline_x", "value": selected_result.baseline_x},
                {"item": "baseline_y_raw", "value": selected_result.baseline_y_raw},
                {"item": "point2_x", "value": selected_result.point2_x},
                {"item": "point2_y_raw", "value": selected_result.point2_y_raw},
                {"item": "point3_x", "value": selected_result.point3_x},
                {"item": "point3_y_raw", "value": selected_result.point3_y_raw},
                {"item": "point4_x", "value": selected_result.point4_x},
                {"item": "point4_y_final", "value": selected_result.point4_y_final},
                {"item": "line1", "value": f"y = {selected_result.m1:.6f}x + {selected_result.b1:.6f}"},
                {"item": "line2", "value": f"y = {selected_result.m2:.6f}x + {selected_result.b2:.6f}"},
                {"item": "ratio1 (%)", "value": selected_result.ratio1},
                {"item": "ratio2 (%)", "value": selected_result.ratio2},
                {"item": "ratio3 (%)", "value": selected_result.ratio3},
                {"item": "area1_abs", "value": selected_result.area1_abs},
                {"item": "area2_abs", "value": selected_result.area2_abs},
                {"item": "area3_abs", "value": selected_result.area3_abs},
            ])
            st.dataframe(detail_df, use_container_width=True)

    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました: {e}")
else:
    st.info("左または上のアップロード欄から Excel ファイルを選んでください。")
    st.markdown(
        """
        ### 使い方
        1. Excel ファイルをアップロード
        2. sidebar で各探索範囲を設定
        3. **全サンプルを解析** をクリック
        4. 結果一覧と個別グラフを確認
        5. CSV / Excel で結果を保存
        """
    )
