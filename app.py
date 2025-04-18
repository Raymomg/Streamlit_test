
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 데이터 불러오기
load_df = pd.read_csv("input.csv")
raw_df = load_df.copy()
raw_df.head(5)
X = raw_df[
    [
        "로그 원수 탁도",
        "원수 pH",
        "원수 알칼리도",
        "원수 전기전도도",
        "원수 수온",
        "3단계 원수 유입 유량",
        "3단계 침전지 체류시간",
    ]
]
y = raw_df["로그 응집제 주입률"]
Xt, Xts, yt, yts = train_test_split(X, y, test_size=0.2, shuffle=False)

def main():


    st.markdown("## Raw Data")
    st.dataframe(raw_df)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## Select")
    column = st.selectbox("타겟변수를 선택하세요", raw_df.columns)
    st.dataframe(raw_df[column])

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("## MultiSelect")
    cols = st.multiselect("복수의 컬럼을 선택하세요", raw_df.columns)
    filtered_raw_df = raw_df.loc[:, cols]
    st.dataframe(filtered_raw_df)

    max_depth = st.slider("max_depth:",min_value=0, max_value=20)
    n_estimator = st.slider("n_estimator:", min_value=0, max_value=500)
    learning_rate = st.slider("learning_rate:", min_value=0.0, max_value=1.0)
    subsample = st.slider("subsample:", min_value=0.0, max_value=1.0)


    #XGBoost모델개발
    model = XGBRegressor(
        random_state=2,
        n_jobs=-1,
        max_depth=max_depth,
        n_estimators=n_estimator,
        learning_rate=learning_rate,  
        subsample=subsample,
        min_child_weight=1
    )


    model.fit(Xt, yt)


    #모델성능평가
    yt_pred = model.predict(Xt)
    yts_pred = model.predict(Xts)

    mse_train = mean_squared_error(10**yt, 10**yt_pred)
    mse_test = mean_squared_error(10**yts, 10**yts_pred)
    st.write(f"학습 데이터 MSE: {mse_train}")
    st.write(f"테스트 데이터 MSE: {mse_test}")

    r2_train = r2_score(10**yt, 10**yt_pred)
    r2_test = r2_score(10**yts, 10**yts_pred)
    st.write(f"학습 데이터 R2: {r2_train}")
    st.write(f"테스트 데이터 R2: {r2_test}")


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(Xt["로그 원수 탁도"], yt, s=3, label="학습 데이터 (실제)")
    ax.scatter(Xt["로그 원수 탁도"], yt_pred, s=3, label="학습 데이터 (예측)", c="r")
    ax.grid()
    ax.legend(fontsize=13)
    ax.set_xlabel("로그 원수 탁도")
    ax.set_ylabel("로그 응집제 주입률")
    ax.set_title(
        rf"학습 데이터  MSE: {round(mse_train, 4)}, $R^2$: {round(r2_train, 2)}",
        fontsize=18,
    )

    ax = axes[1]
    ax.scatter(Xts["로그 원수 탁도"], yts, s=3, label="테스트 데이터 (실제)")
    ax.scatter(Xts["로그 원수 탁도"], yts_pred, s=3, label="테스트 데이터 (예측)", c="r")
    ax.grid()
    ax.legend(fontsize=13)
    ax.set_xlabel("로그 원수 탁도")
    ax.set_ylabel("로그 응집제 주입률")
    ax.set_title(
        rf"테스트 데이터  MSE: {round(mse_test, 4)}, $R^2$: {round(r2_test, 2)}",
        fontsize=18,
    )
    st.pyplot(fig)

        

if __name__ == "__main__":
    main()
