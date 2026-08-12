data_dictionary = {
    "limit_bal": "Credit limit assigned to the customer",
    "sex": "Customer gender code",
    "education": "Education level code",
    "marriage": "Marital status code",
    "age": "Customer age",
    "pay_0": "Repayment status for most recent month",
    "pay_2": "Repayment status for previous month",
    "pay_3": "Repayment status for 3 months prior",
    "pay_4": "Repayment status for 4 months prior",
    "pay_5": "Repayment status for 5 months prior",
    "pay_6": "Repayment status for 6 months prior",
    "bill_amt1": "Bill amount for most recent month",
    "bill_amt2": "Bill amount for previous month",
    "bill_amt3": "Bill amount for 3 months prior",
    "bill_amt4": "Bill amount for 4 months prior",
    "bill_amt5": "Bill amount for 5 months prior",
    "bill_amt6": "Bill amount for 6 months prior",
    "pay_amt1": "Payment amount for most recent month",
    "pay_amt2": "Payment amount for previous month",
    "pay_amt3": "Payment amount for 3 months prior",
    "pay_amt4": "Payment amount for 4 months prior",
    "pay_amt5": "Payment amount for 5 months prior",
    "pay_amt6": "Payment amount for 6 months prior",
    "default_flag": "Whether the customer defaulted on payment",
}


for column, description in data_dictionary.items():
    print(f"{column}: {description}")