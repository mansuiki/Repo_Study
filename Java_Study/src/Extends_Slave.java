public class Extends_Slave extends Extends_Human {

    @Override
    public void sleep() {
        System.out.println("노예는 잠을 자지 못한다");
        hp -= 10;
    }

    @Override
    public void eat() {
        super.eat();
        System.out.println("노예는 많이 먹지 못한다");
        hp += 5;
    }

    @Override
    public void active() {
        System.out.println("노예는 열심히 살아야 한다");
        hp = 0;
    }

    @Override
    public void work() {
        System.out.println("노예야 일해라!");
    }
}
