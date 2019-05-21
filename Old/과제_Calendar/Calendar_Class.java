import java.util.Calendar;

public class Calendar_Class {
    Calendar cal = Calendar.getInstance(); // 캘린더 사용
    int year; // 연도
    int month; // 월
    int day; // 일
    int whatDay; // 요일

    public Calendar_Class () {
    }

    public Calendar_Class (String whatYouWantToKnow) {
        if (whatYouWantToKnow == "past") {
            yesterday();
            print();
        }
        else if (whatYouWantToKnow == "now") {
            now();
            print();
        }
        else if (whatYouWantToKnow == "future") {
            tomorrow();
            print();
        }
        else {
            System.out.println("past, now, future 중 하나를 입력하세요");
        }
    }

    void print() {
        System.out.printf("%d년 %d월 %d일 ", this.year, this.month, this.day);
        String[] week = {" ", "일", "월", "화", "수", "목", "금", "토"};
        System.out.printf("%s요일\n", week[this.whatDay]);

    }
    void now() {
        this.year = cal.get(Calendar.YEAR);
        this.month = cal.get(Calendar.MONTH) + 1;
        this.day = cal.get(Calendar.DAY_OF_MONTH);
        this.whatDay = cal.get(Calendar.DAY_OF_WEEK);
    }

    void tomorrow() {
        this.year = cal.get(Calendar.YEAR);
        this.month = cal.get(Calendar.MONTH) + 1;
        this.day = cal.get(Calendar.DAY_OF_MONTH) + 1;
        this.whatDay = cal.get(Calendar.DAY_OF_WEEK) +1 ;
    }

    void yesterday() {
        this.year = cal.get(Calendar.YEAR);
        this.month = cal.get(Calendar.MONTH) + 1;
        this.day = cal.get(Calendar.DAY_OF_MONTH) - 1;
        this.whatDay = cal.get(Calendar.DAY_OF_WEEK) - 1;
    }
}
