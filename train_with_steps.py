import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 環境を作成
env = gym.make("HalfCheetah-v4")

# 10,000ステップごとにモデルを自動保存する設定
checkpoint_callback = CheckpointCallback(
    save_freq=10000, 
    save_path="./models_standard/", 
    name_prefix="cheetah_model"
)

print("段階的に学習を進めます（合計50,000ステップ）...")
model = PPO("MlpPolicy", env, verbose=1)

# 学習開始
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# 最終モデルの保存
model.save("./models/cheetah_model_final")
print("すべての学習と保存が完了しました！")