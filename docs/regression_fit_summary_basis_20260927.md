# OLS fit summary and nested comparison basis (2026-09-27)

The library already returns OLS coefficients, residuals, HC covariance, and contrast tests. A consumer still computes centered \(R^2\), adjusted \(R^2\), and the classical nested-model \(F\) statistic outside the library. These quantities are general OLS summaries, so this change adds a Rust comparison for a full design and specified dropped columns. It does not define a study's model terms or conditional contrasts.

For a design with an intercept, the centered total sum of squares is \(\sum_i(y_i-\bar y)^2\), \(R^2=1-\mathrm{SSE}/\mathrm{SST}\), and adjusted \(R^2=1-(1-R^2)(n-1)/(n-k)\). Apply the last formula with each design's own \(k\), so the result includes adjusted \(R^2\) for both the full and reduced models. Pennsylvania State University's STAT 501 Lesson 5, “Coefficient of Determination, R-squared, and Adjusted R-squared,” states both formulas. The same course's Lesson 8, “Categorical Predictors,” gives the general linear \(F=[(\mathrm{SSE}_R-\mathrm{SSE}_F)/(df_R-df_F)]/[\mathrm{SSE}_F/df_F]\). These are classical OLS quantities; HC Wald tests remain separate. Both source passages were opened on 2026-09-27.

Implementation contracts: identify an intercept column in both designs before returning centered \(R^2\) or adjusted \(R^2\); reject zero SST, non-finite sums, incompatible nested degrees of freedom, and a reduced SSE below full SSE beyond numerical tolerance. Derive the reduced design from the full design and a sorted, unique list of dropped column indices, so the comparison uses the same response rows and a genuinely nested model. Test exact synthetic fits, no-intercept designs, degenerate responses, and invalid column lists through Rust and Python.

## Sources

Pennsylvania State University, Department of Statistics. (n.d.). *Lesson 5: Multiple linear regression*. STAT 501: Regression methods. https://online.stat.psu.edu/stat501/Lesson05

Pennsylvania State University, Department of Statistics. (n.d.). *Lesson 8: Categorical predictors*. STAT 501: Regression methods. https://online.stat.psu.edu/stat501/Lesson08
