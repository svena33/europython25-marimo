from EUP25_Marimo_D2 import load_model, predict_sales, surpress_warnings

surpress_warnings()

sales_data = [595, 989, 333, 506, 864, 1396, 806]
event_data = [1, 0, 0, 0, 0, 0, 1]

predicted_sales_next_day = predict_sales(load_model(), sales_data, event_data)

print(f"Predicted sales for the next day: {predicted_sales_next_day}")
