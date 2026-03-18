"""Prompts for the Calculator Agent (Phase 2 stub)."""

def get_calculator_stub_response(aquifer_context: str, selected_method: str) -> str:
    """
    Generate stub response for calculation requests.

    Args:
        aquifer_context: Aquifer type from state
        selected_method: Method name from state

    Returns:
        Formatted stub response
    """
    method_text = f"the {selected_method}" if selected_method and selected_method != "" else "your selected"
    aquifer_text = f"{aquifer_context} aquifer" if aquifer_context != 'unknown' else "your aquifer"

    return f"""**Calculation Module - Coming in Phase 2 (BTP-2)**

I understand you want to perform pumping test analysis for {aquifer_text} using {method_text} method.

**Phase 2 will include:**
- Curve-fitting analysis using nonlinear least squares
- Automatic parameter estimation (Transmissivity T, Storativity S, Hydraulic Conductivity K)
- Diagnostic plots (time-drawdown, residuals)
- Goodness-of-fit metrics and confidence intervals
- Data validation and quality checks

**For now, I can help you with:**
- Understanding the theoretical basis of {method_text} method
- Explaining how to prepare and format your pumping test data
- Discussing which analysis method is most appropriate for {aquifer_text}
- Reviewing the assumptions and limitations of different methods

Would you like me to explain the calculation methodology, or help you understand the theory better?"""


CALCULATION_EXPLAINER = """To perform pumping test analysis, you'll need the following data:

**Required Data:**
1. **Time-drawdown data**:
   - Time since pumping started (minutes or hours)
   - Drawdown measurements (meters or feet)
   - At least 10-20 data points for good analysis

2. **Test configuration**:
   - Pumping rate (Q) - constant during test
   - Distance from pumping well to observation well (r)
   - Aquifer thickness (b) if confined

3. **Site information**:
   - Aquifer type (confined, unconfined, leaky)
   - Preliminary estimates of hydraulic properties (if available)

**Data Format Example (CSV):**
```
time_minutes,drawdown_meters
1,0.15
2,0.28
5,0.52
10,0.75
20,1.02
...
```

**Next Steps:**
1. Prepare your data in this format
2. When Phase 2 is released, you'll be able to upload it directly
3. The system will automatically fit the appropriate method and estimate parameters

Would you like me to explain the theory behind how the curve-fitting will work?"""
