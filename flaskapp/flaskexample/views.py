# Create my flask app
from flask import render_template
from flask import request
from flaskexample import app
import pandas as pd
import numpy as np  


### READ IN DATA ###

# remote working directory should be /home/ubuntu/application

# open full listings data set and depreciation summary
listings_data = pd.read_csv('./flaskexample/data/listings5.csv')
depr_summary = pd.read_csv('./flaskexample/data/depreciation_by_model_2.csv')
make_model_list = pd.read_csv('./flaskexample/data/make_model_list_sorted.csv')
model_counts_filtered = pd.read_csv('./flaskexample/data/model_counts_filtered.csv')

### PRE-PROCESS DATA ###

# exclude models older than ___ year, with fit quality below ___ R2 
newerthan = 1995
betterthan = 0.67
nomorethan = 300
listings_data_filtered = listings_data[listings_data['Year'] > newerthan]
depr_summary_filtered = depr_summary[depr_summary['Fit_age_R2'] > betterthan]
make_model_list_filtered = make_model_list[:300]

# create dropdown menu containing make/model sorted by frequency
model_counts = listings_data.groupby('Model').count().iloc[:,1].to_frame().rename(columns={'Make':'Counts'}).sort_values(by = 'Counts', ascending = False)

# view all make/model combinations with minimum R2 fit quality
model_counts_filtered = depr_summary_filtered.merge(model_counts.reset_index(), on='Model', how='left')

### CREATE USER INPUTS ###
make_choices = model_counts_filtered['Make'].unique()


# create global input_make
global_input_make = None
global makes
makes = ['Acura', 'Audi', 'BMW', 'Buick', 'Cadillac', 'Chevrolet',
         'Chrysler', 'Dodge', 'Ford', 'GMC', 'Honda', 'Hyundai', 'INFINITI',
         'Jaguar', 'Jeep', 'Kia', 'Land Rover', 'Lexus', 'Lincoln', 'MAZDA',
         'MINI', 'Maserati', 'Mercedes-Benz', 'Mitsubishi', 'Nissan',
         'Porsche', 'smart', 'Subaru', 'Toyota', 'Volkswagen', 'Volvo']



### DEFINE FUNCTIONS ###

def make_dropdown(input_make):
    # global makes
    models = np.unique(model_counts_filtered[model_counts_filtered['Make'] == makes[int(input_make)]]['Model'].tolist())   
    models.sort()
    make_name = makes[int(input_make)]
    
    dropdown_html = "<select id=\"input_model\" name=\"input_model\">\n"
    for i, model in enumerate(models):
            dropdown_html += "<option value=\"{}\">{}</option>\n".format(i, model)
    dropdown_html += "</select>\n"
    return dropdown_html, make_name



### DEFINE FLASK FUNCTIONS ###

@app.route('/')
@app.route('/index')
@app.route('/input')

def index():
    return render_template("index.html")



@app.route('/models') #, methods = ['GET', 'POST'])           # trying to avoid global variable
def get_models():
    global makes
    input_make = request.args.get('input_make')
    global global_input_make 
    global_input_make = input_make
    make_name = makes[int(input_make)]
    (dropdown_html, make_name) = make_dropdown(input_make)
    # global_input_make = None
    # if request.method == 'POST':                            # trying to avoid global variable
    #     input_make = request.form.get('input_make')
    #     return redirect(url_for('output', input_make=input_make))
    return render_template("models.html", input_make=input_make, dropdown_html=dropdown_html, make_name=make_name)



@app.route('/output')
def output():
    global makes
    # pull 'input_model' from input field and store it
    input_model = request.args.get('input_model')
    # input_make = request.args.get('input_make', None)       # trying to avoid global variable
    input_model_name = model_counts_filtered[model_counts_filtered['Make'] == makes[int(global_input_make)]]['Model'].sort_values().iloc[int(input_model)]
    input_model_filename = str(input_model_name) + '.png'
    
    return render_template("output.html", input_model=input_model, input_model_filename=input_model_filename)



@app.route('/random')
def random():
    # generate random entry
    import numpy.random as npr
    selected = model_counts_filtered.sort_values('Counts', ascending=False)[:50]
    random_model = selected.iloc[npr.randint(0,len(selected))][2]
    random_model_filename = str(random_model) + '.png'
    # model_name = get_model_name(input_make, input_model)
    # models_list = models[int(input_make)]
    # model_name = models_list[int(input_model)]
    return render_template("random.html", random_model_filename=random_model_filename)



@app.route('/about')
def about():
    return render_template("about.html")


