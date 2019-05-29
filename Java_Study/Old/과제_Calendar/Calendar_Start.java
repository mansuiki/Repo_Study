
public class Calendar_Start {
    public static void main(String[] args) {
        Calendar_Class cal = new Calendar_Class("now");

        cal.yesterday();
        cal.print();

        cal.tomorrow();
        cal.print();

    }
}
