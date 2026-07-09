import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.sandbox import CodeSandbox

# Adjust this path to wherever you saved sample_sales.csv
data_path = r"C:\Users\Janvi\Documents\sample_sales.csv"

sandbox = CodeSandbox(data_path=data_path)

test_code = """
region_revenue = df.groupby('region')['revenue'].sum()
region_revenue.plot(kind='bar', title='Revenue by Region', color='skyblue')
plt.ylabel('Revenue')
plt.xlabel('Region')
"""

result = sandbox.run(test_code)

print("Success:", result["success"])
print("Stdout:", result["stdout"])
print("Stderr:", result["stderr"])

if result["chart_bytes"]:
    with open("test_output_chart.png", "wb") as f:
        f.write(result["chart_bytes"])
    print("Chart saved as test_output_chart.png — open it to check!")


# Test 2: Intentionally broken code
print("\n--- Testing error handling ---")
broken_code = """
region_revenue = df.groupby('regionn')['revenue'].sum()  # typo: 'regionn'
region_revenue.plot(kind='bar')
"""

result2 = sandbox.run(broken_code)
print("Success:", result2["success"])
print("Stderr:", result2["stderr"])