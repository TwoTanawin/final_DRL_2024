

model_path = "/home/two-asus/Documents/ait/drl/final_DRL_2024/report/A2C_MLP_mainEnv_timmer_RobotPath/model/80000.zip"


text = model_path.split("/")

last_text = text[-1].split(".")

print(text)
print(last_text[0])