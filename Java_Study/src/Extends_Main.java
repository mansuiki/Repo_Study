public class Extends_Main {
    public static void main(String[] args) {
        Extends_Human H = new Extends_Human();
        H.eat();
        H.sleep();
        H.active();

        System.out.println("\n\n\n\n\n");

        Extends_Slave S = new Extends_Slave();
        S.eat();
        S.sleep();
        S.active();
        S.work();
    }
}
