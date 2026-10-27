package demo;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
class CalculatorTest {
  @Test void addWorks() { assertEquals(2, Calculator.add(1, 1)); }
}
