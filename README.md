# Neural Network to predict the likihood of empoloyee leaving the company

This will create a neural network that HR can use to predict whether employees are likely to leave the company.  
It will also predict the department that best fits each employee.  
Branched neural network will be used for making the predictions.  
StandardScaler will be used to fit the scaler to the training data, and then transform both the training and testing data.  
OneHotEncoder will be used for the department column, then fit the encoder to the training data and use it to transform both the training and testing data.  
OneHotEncoder will be used for the attrition column, then fit the encoder to the training data and use it to transform both the training and testing data.  

Build neural network - 
Input layer:
<code>input_layer = layers.Input(shape=column_nums, name='input_features')</code>

### Create at least two shared layers
<code>shared_layer1 = layers.Dense(32, activation='relu')(input_layer)</code>
<code>ishared_layer2 = layers.Dense(16, activation='relu')(shared_layer1)</code>
<code>ishared_layer3 = layers.Dense(12, activation='relu')(shared_layer2)</code>

### Create a branch for Attrition
<code>attr_hidden = layers.Dense(8, activation='relu', name='attr_hidden')(shared_layer3)</code>
### Create the output layer
<code>attr_output = layers.Dense(2, activation='softmax',name='attr_output')(attr_hidden)</code>


### Create the hidden layer
<code>dept_hidden = layers.Dense(8, activation='relu', name='dept_hidden')(shared_layer3)</code>

### Create the output layer
<code>dept_output = layers.Dense(3, activation='softmax', name='dept_output')(dept_hidden)</code>

### Create the model
<code>model = Model(inputs=input_layer, outputs=[attr_output, dept_output])</code>
</code>model = Model(inputs=input_layer, 
              outputs=[attr_output, dept_output],
              name='predict_attrition_model')</code>

### Compile the model
</code>model.compile(optimizer='adam',
              loss={'attr_output': 'categorical_crossentropy',
                    'dept_output': 'categorical_crossentropy'},
              loss_weights={'attr_output': 1.0, 'dept_output': 2.0, },
              metrics={'attr_output': 'accuracy',
                       'dept_output': 'accuracy'}
              )</code>
              
### Train the model
</code>model.fit(
    X_train_scaled,
    {'attr_output': y_attr_train, 'dept_output': y_dept_train},
    validation_data=(X_test, {'attr_output': y_attr_test, 'dept_output': y_dept_test}),
    epochs=15,
    batch_size=5,
    validation_split=0.2,
    verbose=1
)</code>

### Evaluate the model with the testing data
</code>test_results = model.evaluate(X_test_scaled, {'attr_output': y_attr_test, 'dept_output': y_dept_test}, verbose=1).</code>

Results:
Attrition Accuracy: 0.84
Department Accuracy: 0.63


Following steps might be helpful in improving model's perfromance:
- Select additional features
- Select features that have greater impact on outcomes (subject matter expertise is necessary)
- Increase complexity of the model by increasing number of layers and/or number neurons.
- Experiment with different learning rates, batch sizes, and optimizers.
- If possible, generate synthetic data to increase the size of dataset.

