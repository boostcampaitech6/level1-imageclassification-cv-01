import torch

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU가 사용 가능합니다.")
    print("GPU 정보:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU를 사용할 수 없습니다. CPU로 진행합니다.")

# 랜덤한 텐서 생성 및 GPU로 전송
x = torch.rand(3, 3).to(device)

# GPU에서 텐서 계산 수행
y = x + x

# 결과 출력
print("원본 텐서:")
print(x)
print("\n계산 결과:")
print(y)
