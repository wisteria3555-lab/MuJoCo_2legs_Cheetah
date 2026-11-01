import gymnasium as gym
from stable_baselines3 import PPO
import time

# 1. 読み込むモデルのファイルパスを指定
# ※ここを変更することで、「学習初期」「10万ステップ目」「最終結果」などの切り替えができる
model_path = "./models_standard/cheetah_model_final" 
# 例: 途中のモデルを見たい場合は "./models_custom/cheetah_stable_100000_steps.zip" などに書き換える
# ほかの学習モデルを見たい場合は "./models_standard/cheetah_model_final.zip" などに書き換える

# 2. 描画ありの環境を作成
# render_mode="human" を指定することで、シミュレーションの3Dウィンドウを立ち上げる
env = gym.make("HalfCheetah-v4", render_mode="human")

# 3. 学習済みAIモデルのロード
try:
    # 保存されたzipファイルからAIのニューラルネットワークモデルを読み込む
    model = PPO.load(model_path)
    print(f"モデルをロードしました: {model_path}")
except Exception as e:
    # 万が一ファイルが見つからない場合はエラーメッセージを出し、ランダムな動きで代用する
    print(f"モデルが見つかりません。ランダムな初期状態（学習前）で動かします。エラー: {e}")
    model = None

# 4. 環境のリセットと変数の初期化
# シミュレーションを初期状態に戻し、初期観測値と付加情報を受け取る
observation, info = env.reset()
step_count = 0  # 経過ステップ数をカウントする変数

# 5. 推論と描画のメインループ
while True:
    # AIモデルが存在する場合は、現在の状態から「次に取るべき最適な行動」を予測させる
    if model is not None:
        # deterministic=True にすることで、最も確率の高い安定した行動を選ぶ
        action, _states = model.predict(observation, deterministic=True)
    else:
        # モデルがない場合は、環境の行動空間から完全にランダムな動きをサンプリングする
        action = env.action_space.sample()
        
    # 決定した行動を環境に渡し、1ステップ分シミュレーションを進める
    observation, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    
    # 6. 自動リセット（ループ再生）の判定
    # チーターが転倒などしてエピソードが終了（terminated/truncated）したか、
    # あるいは同じ場所でずっと走り続けて画面外に行ってしまわないよう、約300ステップで強制リセットする
    if terminated or truncated or step_count > 300:
        observation, info = env.reset()  # 初期位置に戻す
        step_count = 0                   # カウントをリセット
        
    # 7. 描画スピードの調整
    # パソコンが速すぎて動作がコマ送りになりすぎないよう、わずかにウェイト（0.01秒）を入れる
    time.sleep(0.01)