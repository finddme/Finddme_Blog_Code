'''
# 클래스 상속

- 상속은 객체지향 프로그램의 특성 중 하나이다.
  객체지향 프로그램인 파이썬에서는 상속을 통해
  다른 클래스의 멤버변수와 메소드를 물려받아 사용할 수 있다.
  상속이라는 기능으로 인해 소스코드가 단축될 수 있고, 프로그램이 한층 더 계층적으로 구성될 수 있다.
- 기존의 클래스가 가진 속성을 모두 포함하는 또 다른 클래스를 만들 경우, 기존 클래스를 상속받는다.
- 상속을 하면 클래스 간에 부모/자식 관계가 형성된다.
  상속을 받은 클래스가 자식 클래스이고, 상속을 해준 클래스가 부모 클래스이다.
- 자식 클래스의 멤버변수나 함수는 부모클래스에서 사용할 수 없다.
'''
# Dessert라는 클래스를 만들어보자.
# Dessert 클래스는 sort와 flavor를 속성으로 갖는다.
# 그리고 order이라는 함수를 정의하여 어떤 디저트를 주문할지 출력하도록 한다.
class Dessert:
    def __init__(self, sort, flavor):
        self.sort = sort
        self.flavor = flavor
    def order(self):
        print("후식은",self.flavor, self.sort , "으로 주세요.")
    def taste(self):
        print("이 집", self.sort, "잘하네")

dessert1 = Dessert("도넛", "글레이즈드") # dessert1이라는 인스턴스를 만들어서 클래스를 사용한다.
dessert1.order() # 인스턴스 dessert1에 order함수를 불러와서 실행한다.
dessert1.taste()

'''
상속받기
'''
# 클래스 이름 뒤에 괄호를 넣어 거기에 상속하고자 하는 클래스의 이름을 넣어준다.

class Beverage(Dessert):
    def __init__(self, sort, flavor, syrup):
        Dessert.__init__(self, sort, flavor) # Dessert 클래스 생성자의 멤버변수를 받는다.
        self.syrup = syrup
    def order_beverage(self):
        print("음료는", self.flavor, self.sort, self.syrup, "으로 주세요.")

beverage1 = Beverage("스무디", "딸기","no-syrup") # beverage1라는 인스턴스를 만든다.
beverage1.order_beverage()
beverage1.taste() # 부모 클래스의 함수를 써보자