import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. カスタム報酬ラッパーの定義
class StrictHalfCheetahWrapper(gym.Wrapper):
    #チーターが転倒や過度な反り返りをしたときに、
    #通常の報酬からペナルティを与えるためのカスタム環境クラス
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        # AIが選んだ行動を環境に反映させ、次の状態・報酬・終了フラグ等を受領
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # MuJoCoの内部物理データから、チーターの胴体の傾きを取得
        root_angle = self.unwrapped.data.qpos[2]
        
        # 傾きが0.5ラジアンを超えて反りかえったり倒れそうになった場合
        if abs(root_angle) > 0.5:  
            reward -= 5.0  # 強いペナルティを与える
            
        return observation, reward, terminated, truncated, info

# 2. 環境の準備とラップ
# Gymnasiumから公式のHalfCheetah環境をロードする
base_env = gym.make("HalfCheetah-v4")

# 環境を転倒防止ペナルティ付きのラッパーでラップ
env = StrictHalfCheetahWrapper(base_env)

# 3. 途中経過を保存
# 指定ステップ学習が進むごとに、モデルのバックアップを "./models_custom/" フォルダに保存する設定
checkpoint_callback = CheckpointCallback(
    save_freq=100000,                  # 保存する間隔（ステップ数）
    save_path="./models_custom/",       # 保存先のフォルダ名
    name_prefix="cheetah_stable"        # 保存されるファイル名のプレフィックス
)

print("走行学習を開始...")

# 4. PPOモデルの初期化
# MlpPolicy: ニューラルネットワークを使って学習する標準的な設定
# ent_coef=0.01: 少しだけランダムな行動（探索）を多めに許可し、新しい走り方の発見を促す
model = PPO("MlpPolicy", env, ent_coef=0.01, verbose=1)

# 5. 学習の実行
# 指定した総ステップ数だけAIに試行錯誤させ、コールバックで定期保存を行う
model.learn(total_timesteps=300000, callback=checkpoint_callback)

# 6. 最終モデルの保存
# 学習がすべて終わった最終的なモデルをファイルとして保存
model.save("./models_custom/cheetah_stable_final")
print("学習が完了しました！ './models_custom/' フォルダを確認してください。")