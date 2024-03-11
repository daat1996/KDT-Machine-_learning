# 클래스 생성 (1) ------------------------------------------------
# - 구성 요소 : 속성 + 메서드  => 모두 없는 클래스
# - 기본 상속: Object  ==> __속성명__, __메서드명__()
# -----------------------------------------------------------------

class A:
    pass
# 클래스 생성 (2) ------------------------------------------------
# - 구성 요소 : 속성 + 메서드  => 인스턴스 변수와 메서드
# - 기본 상속: Object  ==> __속성명__, __메서드명__()
# -----------------------------------------------------------------
class B:
    # 인스턴스 객체 생성 및 속성 초기화 메서드
    def __init__(self, num,name):
        # self로 지정된 힙 메모리 주소에서 부터 속성 저장
        self.num = num
        self.name = name

    # 인스턴스 메서드
    def printInfo(self):
        print(f'num   : {self.num}')
        print(f'name  : {self.name}')


# 클래스 생성 (3) ------------------------------------------------
# - 구성 요소 : 속성 + 메서드  => 인스턴스 변수와 메서드, 클래스 변수
# - 기본 상속: Object  ==> __속성명__, __메서드명__()
# -----------------------------------------------------------------
class C:
    # 클래스 변수 => c 클래스로 생성된 모든 인스터스에서 공유
    #             => 인스턴스 생성 없이 사용 가능
    loc = 'Dauegu'

    # 인스턴스 객체 생성 및 속성 초기화 메서드
    def __init__(self, num, name):
        # self로 지정된 힙 메모리 주소에서 부터 속성 저장
        self.num = num
        self.name = name

    # 인스턴스 메서드
    def printInfo(self):
        print(f'num   : {self.num}')
        print(f'name  : {self.name}')


# 클래스 생성 (4) ------------------------------------------------
# - 구성 요소 : 속성 + 메서드  => 클래스 변수와 메서드
# - 기본 상속: Object  ==> __속성명__, __메서드명__()
# -----------------------------------------------------------------
class DCalc:
    # 클래스 변수 => c 클래스로 생성된 모든 인스터스에서 공유
    #             => 인스턴스 생성 없이 사용 가능
    name = 'CASIO'

    # 클래스 메서드

    def addNum(cls, a, b):
        # self로 지정된 힙 메모리 주소에서 부터 속성 저장
        print(cls)
        return a+b

    def minusNum(cls, a ,b):
        return a-b


# 객체/인스턴스 생성 ----------------------------------------------------
# => 생성 함수: 클래스이름(__init__메서드 매개변수)
# => A(# )
# ------------------------------------------------------------------------
a1=A()
b1=B(100,'BB')
# c1=C(1000,'CCC')


# 객체/인스턴스의 속성/메서드 사용 ---------------------------------------
# => 사용 방법 : 객체/인스턴스 변수명.속성
#               : 객체/인스턴스 변수명.메서드()
# ------------------------------------------------------------------------
print("B 인스턴스 b1의 속성 =>", b1.__dict__)
print("B 인스턴스 b1의 속성과 메서드 =>", b1.__dir__())    # 인스턴스 변수까지 싹 다 들고 온다.
print("B 클래스의 속성과 메서드=>   ", B.__dict__)

# 인스턴스 메서드 사용
# c1.printInfo()

# 인스턴스 속성 사용
# print(c1.name)

# 클래스 속성 사용
print("loc =>", C.loc)

# 인스턴스 메서드는 클래스명으로 사용 불가!! => self 즉 인스턴스 주소 및 정보 없음
# self 들어가있으면 인스턴스 가 있어야함
# C.printInfo()