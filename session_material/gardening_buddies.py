import pulp
import csv

with open('kg_m2_de.csv', 'r') as infile:
    # read the file as a dictionary for each row ({header : value})
    reader = csv.DictReader(infile)
    data = {}
    for row in reader:
        for header, value in row.items():
            try:
                value = float(value)
            except ValueError:
                value = value
            try:
                data[header].append(value)
            except KeyError:
                data[header] = [value]


# Declare problem and whether the objective function is to maximize or minimize
problem = pulp.LpProblem(name="Surface", sense=pulp.LpMinimize)

# Declare menu list and nutritional information
#sodium = [1220, 1510, 330, 0]
# Declare nutritional requirements
nutrition_requirement_dict = {
    "Calories": 2700.0,
    "Protein": 60.0,  # g
    "Carbohydrates": 130,  # g
    "Fat": 10,  # g
    "Fiber": 200.0,
    "Cholesterol": 0.3,
    "C_rda": 0.09,
    # "Sodium" : 7500, # mg
}


# Declare LpVariables for each menu item
LpVariableList = [pulp.LpVariable('{}'.format(
    item), lowBound=0, upBound=3) for item in data['Name']]

# Declare objective: to minimize surface
problem += pulp.lpDot(data['Surface_per_kilo'], LpVariableList)

# Declare constraints
# Assume the daily energy requirements of a man aged 30-49 with low activity
problem += pulp.lpDot(data['Data.Protein'],
                      LpVariableList) >= nutrition_requirement_dict["Protein"]
problem += pulp.lpDot(data['Data.Carbohydrate'],
                      LpVariableList) >= nutrition_requirement_dict["Carbohydrates"]
problem += pulp.lpDot(data['Data.Kilocalories'],
                      LpVariableList) >= nutrition_requirement_dict["Calories"]
problem += pulp.lpDot(data['Data.Fiber'],
                      LpVariableList) >= nutrition_requirement_dict["Fiber"]
problem += pulp.lpDot(data['Data.Fat.Total.Lipid'],
                      LpVariableList) >= nutrition_requirement_dict["Fat"]
problem += pulp.lpDot(data['Data.Vitamins.Vitamin.C'],
                      LpVariableList) >= nutrition_requirement_dict["C_rda"]

# Solve
status = problem.solve()
print(pulp.LpStatus[status])
# Result
print("Result\n")
for menu_item in LpVariableList:
    if menu_item.value():
        print(str(menu_item) + " Ã— " + str((menu_item.value())))

# for nutrient_name, nutrient_value in {"Calories": data['Data.Kilocalories'], "Protein": data['Data.Protein'], "Carbohydrates": data['Data.Carbohydrate'],"Fiber":data['Data.Fiber'],"Fat":data['Data.Fat.Total.Lipid']}.items():
        #print("*{}: {} Reference: {}".format(nutrient_name, str(round(pulp.lpDot(nutrient_value, LpVariableList).value())), nutrition_requirement_dict[nutrient_name]))

print("\nYou need "+str(pulp.value(problem.objective)) +
      " square meters a day to feed You.")
