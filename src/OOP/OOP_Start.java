package OOP;

public class OOP_Start {
    public static void main(String[] args) {
        OOP_Human mansuiki = new OOP_Human(180, 80, 100);
        OOP_Human jobs = new OOP_Human();

        System.out.println("mansuiki sleep status : " + mansuiki.isSleep);
        System.out.println("jobs sleep status : " + jobs.isSleep);

        mansuiki.sleep();
        jobs.sleep();

        mansuiki.homework();
        mansuiki.eat();

        jobs.execerise();

        if (jobs.hp > mansuiki.hp) {
            System.out.println("jobs Win!");
        } else {
            System.out.println("mansuiki Win!");
        }

    }


}
