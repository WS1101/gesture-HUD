package org.example;

public class OneEuroFilter {
    private double minCutoff; // 최소 차단 주파수 (떨림 제거용)
    private double beta;      // 속도 계수 (지연 제거용)
    private double dCutoff;   // 파생 차단 주파수 (보통 1.0)

    private double xPrev;
    private double dxPrev;
    private long tPrev;

    public OneEuroFilter(double minCutoff, double beta) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = 1.0;
        this.xPrev = 0.0;
        this.dxPrev = 0.0;
        this.tPrev = 0;
    }

    // 필터링 함수
    public double filter(double x, long t) {
        if (tPrev == 0) {
            tPrev = t;
            xPrev = x;
            dxPrev = 0.0;
            return x;
        }

        // 주파수 계산
        double te = (t - tPrev) / 1000.0; // 초 단위 변환
        // 신호가 너무 빨리 들어오면(중복) 이전 값 리턴
        if (te <= 0.0) return xPrev;

        tPrev = t;

        // 1. 변화율(속도) 계산 및 필터링
        double dx = (x - xPrev) / te;
        double dxHat = exponentialSmoothing(dx, dxPrev, alpha(te, dCutoff));

        // 2. 속도에 따른 컷오프 주파수 조절 (적응형)
        double cutoff = minCutoff + beta * Math.abs(dxHat);

        // 3. 최종 값 필터링
        double xHat = exponentialSmoothing(x, xPrev, alpha(te, cutoff));

        xPrev = xHat;
        dxPrev = dxHat;

        return xHat;
    }

    private double exponentialSmoothing(double x, double prevX, double alpha) {
        return alpha * x + (1.0 - alpha) * prevX;
    }

    private double alpha(double te, double cutoff) {
        double r = 2 * Math.PI * cutoff * te;
        return r / (r + 1.0);
    }

    // 리셋용
    public void reset() {
        tPrev = 0;
        xPrev = 0.0;
        dxPrev = 0.0;
    }
}
