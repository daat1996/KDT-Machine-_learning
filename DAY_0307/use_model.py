## ------------------------------------------------------------------------------
## 모델을 활용한 서비스 제공
## ------------------------------------------------------------------------------
# 모듈 로딩
from joblib import load

# 전역 변수
model_file = '../model/iris_dt.pkl'

# 모델로딩
model=load(model_file)

# 로딩된 모델 확인
print(model.classes_)

# 붓꽃 정보 입력 => 4개 피쳐
datas = input("붓꽃의 꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비 입력:")
if len(datas):
    datas_list = list(map(float, datas.split(',')))
    # print(datas_list)

    # 입력된 정보에 해당하는 품종 알려주기
    # 모델의 predict(2D)
    # 만일 model에 column이 있다면 데이터프레임화 해야 경고가 안뜬다
    pre_iris = model.predict([datas_list])
    proba = max(model.predict_proba([datas_list])[0])*100
    print(f'해당 꽃이 {pre_iris}일 확률은 {proba}% 입니다')


else:
    print('입력된 정보가 없습니다.')