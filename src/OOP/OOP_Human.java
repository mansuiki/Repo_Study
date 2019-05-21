package OOP;

public class OOP_Human {
    boolean isSleep;
    int height;
    int weight;
    int isHandsome;
    int hp;

    public OOP_Human(int height, int weight, int isHandsome ){
        this.height = height;
        this.weight = weight;
        this.isHandsome = isHandsome;

        System.out.printf("Height : %d\nWeight : %d\nisHandsome : %d\n", height, weight, isHandsome);

    }

    public OOP_Human(){
        System.out.println("사람 생성됨");
    }

    void sleep() {
        hp = 3000;
        isSleep = true;
        System.out.println("Sleep Zzzz..");
    }

    void execerise() {
        System.out.println("운동 운동 운동!");
        isSleep = false;
        hp = hp - 100;
    }

    void eat() {
        System.out.println("Eat Eat Eat ...");
        isSleep = false;
        hp = hp + 100;
    }

    void homework() {
        System.out.println("교수 싫어 과제 싫어");
        isSleep = false;
        hp = 0;
    }


}
