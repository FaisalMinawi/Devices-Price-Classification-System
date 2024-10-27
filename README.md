#

\
\
Devices Price Classification System Documentation

## Project Overview

This project involves building a Devices Price Classification System using Python and Spring Boot. The system consists of two main components:

1. **Python Project**: This component is responsible for predicting the prices of devices. The model uses device characteristics to classify the price into different categories (low, medium, high, very high). Sellers can use this model to classify devices based on their specifications.

2. **Spring Boot Project**: This component includes endpoints for managing device records and predicting device prices by calling the Python API for a batch of test cases. Additionally, it stores the predicted prices in a database.

## Project Directory Structure

The project directory is organized as follows:

### Java/Spring Boot Project Directory

1. **com.device.classification**
   - `.mvn/wrapper` - Contains Maven wrapper properties, which are used to ensure that the same Maven version is used across different environments without requiring Maven installation.
   - `.vscode` - VSCode configuration settings, which include specific editor preferences such as automatic build configuration.
   - `src/main/java/com/device/classification` - Contains Java files for Spring Boot:
     - **`Device.java`** - Entity representing a device, which includes attributes like battery power, clock speed, RAM, etc. This entity maps directly to a table in the H2 database.
     - **`DeviceController.java`** - REST controller for handling API requests related to devices. It provides CRUD operations and price prediction features.
     - **`DeviceRepository.java`** - Repository interface for data access. It extends `JpaRepository`, providing methods to interact with the database such as saving, updating, deleting, and retrieving device records.
     - **`MainApplication.java`** - Entry point for running the Spring Boot application. It initializes the Spring context and starts the embedded server.
     - **`ResourceNotFoundException.java`** - Custom exception class used to handle cases where a requested resource (e.g., a device by ID) is not found in the database.
   - `src/main/resources/application.properties` - Configuration file that contains database settings, server port configuration, and other properties that influence application behavior.
   - `.gitignore` - Specifies which files and directories to be excluded from version control, such as compiled Java classes, log files, and temporary files generated by IDEs.

### Python Project Directory

2. **Python API Project**
   - **`model.py`** - Python script to create, train, and evaluate the price classification model. It includes data preparation, feature scaling, model training with hyperparameter tuning, and evaluation using various metrics such as accuracy and confusion matrix.
   - **`predict_price.py`** - Python script to expose a RESTful API for predicting device prices using the trained model. The API accepts device specifications and returns the predicted price category.
   - **`requirements.txt`** - Lists the Python dependencies required to run the project, including libraries like Pandas, NumPy, Scikit-Learn, XGBoost, and Flask.
   - **`train.csv`**** & ****`test.csv`** - Training and testing datasets. The training dataset is used to train the model, and the testing dataset is used to evaluate its performance.

## Key Components and Features

### 1. Python Project

- **Data Preparation**: The dataset provided contains various features such as battery power, number of cores, RAM, screen dimensions, and more, which are used to predict the `price_range`. The `price_range` is the target variable and represents the price category of the device (0 - low, 1 - medium, 2 - high, 3 - very high).

  - **Data Cleaning**: Removed unnecessary columns such as `id` to avoid affecting model training. Handled missing values and performed feature engineering to improve model performance.
  - **Feature Scaling**: Standardized features using `StandardScaler` to ensure that all features contribute equally to the model, especially when they have different units or scales.
  - **Feature Engineering**: Additional features could be engineered, such as combining screen height and width to calculate screen area, or transforming categorical features into binary features using one-hot encoding.

- **Exploratory Data Analysis (EDA)**:

  - Used appropriate charts, such as histograms and scatter plots, to analyze feature distributions and relationships between different variables.
  - **Insights**: For instance, RAM and battery power showed a strong positive correlation with the `price_range`, indicating that higher RAM and battery capacity tend to be associated with higher device prices.

- **Modeling**:

  - **Model Selection**: The project uses an **XGBoost Classifier**, known for its efficiency and accuracy, especially in classification tasks with structured/tabular data.
  - **Hyperparameter Tuning**: Used `GridSearchCV` to identify the best combination of parameters like `n_estimators`, `max_depth`, `learning_rate`, etc., to optimize the model's performance. Cross-validation (`cv=3`) was employed to ensure that the model generalizes well to unseen data.

- **Evaluation**:

  - **Metrics**: The model was evaluated using metrics such as **confusion matrix**, **classification report**, and **accuracy score**. These metrics help understand how well the model distinguishes between different price categories.
  - **Detailed Results**:
    - **Confusion Matrix**:
      ```
      [[100   5   0   0]
       [  3  86   2   0]
       [  0   6  79   7]
       [  0   0  12 100]]
      ```
    - **Classification Report**:
      ```
                  precision    recall  f1-score   support

           0       0.97      0.95      0.96       105
           1       0.89      0.95      0.91        91
           2       0.85      0.86      0.85        92
           3       0.93      0.89      0.91       112
      ```
    accuracy                           0.91       400
    macro avg       0.91      0.91      0.91       400
    weighted avg       0.91      0.91      0.91       400
    ```
    - **Accuracy**: The model achieved an overall **accuracy of 91.25%**.
    - The tuned XGBoost model has been saved successfully.

    ```

- **Model Testing on Selected Devices**:

  - **Testing on 10 Randomly Selected Devices from Test Dataset**:
    ```
          id  battery_power  blue  clock_speed  dual_sim  fc  ...  sc_h  sc_w  talk_time  three_g  touch_screen  wifi
    ```

92    93           1663     1          0.6         1   0  ...    14     1          5        1             1     1
532  533           1171     1          2.8         0   8  ...     8     2          5        1             0     1
314  315           1899     0          0.5         1   0  ...     7     4         11        1             0     1
948  949           1659     0          2.7         0  18  ...    10     9         13        1             0     1
373  374           1543     1          2.6         1   0  ...     6     4          4        1             1     0
697  698           1785     1          2.5         0   0  ...    12     4         18        0             0     1
214  215           1378     1          2.5         1   4  ...     5     1          7        1             0     0
465  466            983     1          1.6         0   7  ...    14     0          4        1             1     0
907  908            854     1          2.6         1   1  ...    12     5         18        1             1     1
342  343           1850     0          2.2         1   9  ...     8     1         10        1             1     1
\`\`\`

- **Predictions for Selected Devices**:
  ```
  Device ID 93.0: Predicted Price Range -> 2
  ```

Device ID 533.0: Predicted Price Range -> 2
Device ID 315.0: Predicted Price Range -> 2
Device ID 949.0: Predicted Price Range -> 2
Device ID 374.0: Predicted Price Range -> 2
Device ID 698.0: Predicted Price Range -> 1
Device ID 215.0: Predicted Price Range -> 3
Device ID 466.0: Predicted Price Range -> 0
Device ID 908.0: Predicted Price Range -> 1
Device ID 343.0: Predicted Price Range -> 2
\`\`\`

- **Model Optimization**:

  - The model was further refined by adjusting the learning rate and incorporating a more comprehensive feature set to minimize overfitting and improve generalizability.

- **Model Persistence**:

  - The final model was saved using **Pickle** (`device_price_model.pkl`) to facilitate reusability without retraining.

- **Endpoint**: The Python script provides a RESTful API (`/predict_price`) using Flask. This API accepts a JSON payload containing the specifications of a device and returns the predicted price range.

### 2. Spring Boot Project

- **Entities**:

  - **`Device`**: A Java entity class representing a device with attributes like `batteryPower`, `clockSpeed`, `RAM`, etc. The entity is annotated with `@Entity` and maps to a table in the H2 database. The field `predictedPriceRange` stores the predicted price value returned by the Python API.

- **Endpoints**:

  - **POST /api/devices**: Adds a new device record to the database. Accepts a JSON request body with device attributes and persists it using the `DeviceRepository`.
  - **GET /api/devices/{id}**: Retrieves the details of a specific device by ID. If the device does not exist, a `ResourceNotFoundException` is thrown, and a `404 Not Found` response is returned.
  - **POST /api/predict/{deviceId}**: Calls the Python API to predict the price range for the specified device based on its attributes. The predicted price is then saved in the `predictedPriceRange` field of the `Device` entity.
    - **Implementation Details**: Uses **`RestTemplate`** to make an HTTP POST request to the Python API. The response from the Python API is parsed using **Jackson** to extract the predicted price and update the device.
    - **Transaction Management**: This endpoint is marked as `@Transactional` to ensure data consistency—if an error occurs while calling the Python API, the transaction will roll back, preventing partial updates.
  - **POST /api/predict/batch\_all**: Predicts the price range for all devices in the database (batch operation for the first 10 records). This endpoint demonstrates how the system can efficiently handle batch predictions.
    - **Error Handling**: If an error occurs while predicting a particular device, it is logged, and the system continues processing the remaining devices.

- **Database Configuration**:

  - The Spring Boot project uses an **H2 in-memory database** for development and testing. The configuration for the H2 database is provided in `application.properties`. This setup allows rapid testing without the need for an external database.
  - **JPA and Hibernate** are used for object-relational mapping, simplifying data access and management.
  - **Database Migration**: In a production scenario, tools like **Flyway** or **Liquibase** can be integrated to manage database schema changes effectively.

## Installation and Setup

### Requirements

- **Java 17**: Required for running the Spring Boot application.
- **Python 3.9+**: Required for running the Python prediction API.
- **Maven**: Used for managing Java dependencies and building the Spring Boot project.
- **Dependencies**:
  - **Spring Boot JPA**: For database operations.
  - **Spring Boot Web**: For creating RESTful web services.
  - **Spring Boot Validation**: To validate input data.
  - **MySQL Connector**: Used if switching from H2 to MySQL for persistent storage.
  - **H2 Database**: An in-memory database for development and testing purposes.
  - **XGBoost, Pandas, NumPy**: Python libraries used for machine learning and data manipulation.

### Steps to Run

1. **Clone the Repository**: Clone the project repository using `git clone`.
2. **Backend (Spring Boot) Setup**:
   - Navigate to the Spring Boot project directory.
   - Run `mvn install` to install the required dependencies and compile the project.
   - Run `MainApplication` to start the Spring Boot application. By default, the app will run on `http://localhost:8080`.
3. **Python API Setup**:
   - Navigate to the Python project directory.
   - Install the required Python dependencies using `pip install -r requirements.txt`.
   - Run `predict_price.py` to start the Python API. By default, it will run on `http://localhost:8000`.
4. **Testing Endpoints**:
   - Use a tool like **Postman** or **Insomnia** to test the API endpoints.
   - For example, to add a device, make a `POST` request to `/api/devices` with a JSON payload containing the device's specifications.
   - To predict the price range of a device, use the `/api/predict/{deviceId}` endpoint.

## Evaluation and Best Practices

- **Code Quality**:

  - Ensure that code is modular, with each function and class having a single responsibility (following the **Single Responsibility Principle**).
  - **Naming Conventions**: Use meaningful variable and method names that clearly describe their purpose. Consistent naming conventions improve code readability and maintainability.
  - **Comments and Documentation**: Each algorithm or concept applied has corresponding comments to articulate the rationale behind decisions. Additional documentation is provided where complex logic is involved.
  - **Exception Handling**: Use custom exceptions like `ResourceNotFoundException` to handle specific error scenarios. Always provide informative error messages that help in debugging.

- **Error Handling**:

  - The Spring Boot project uses **global exception handling** to manage exceptions and provide standardized error responses. The `GlobalExceptionHandler` class (not detailed here) can be used to capture exceptions thrown by any controller and provide a consistent JSON response structure.

- **Security**:

  - Basic security mechanisms, such as input validation and parameterized queries, should be used to prevent SQL Injection and other attacks.
  - In a production environment, **Spring Security** can be added to secure API endpoints and manage authentication and authorization.

- **Testing**:

  - **Unit Testing**: Implement unit tests for both Java and Python projects. Use **JUnit** for Java and **Pytest** for Python to validate individual components and functionalities.
  - **Integration Testing**: Integration tests ensure that the Spring Boot application interacts correctly with the Python API. Mock frameworks such as **Mockito** can be used to simulate interactions between components.
  - **End-to-End Testing**: The `/api/predict/batch_all` endpoint is useful for verifying end-to-end functionality, where multiple devices are processed, and their predicted prices are validated.

- **Logging and Monitoring**:

  - **Logging**: Use a logging framework such as **SLF4J with Logback** for consistent logging in the Spring Boot application. Log important events, especially errors and warnings, to help with debugging and monitoring.
  - **Monitoring**: In a production environment, tools like **Prometheus** and **Grafana** can be used to monitor the health and performance of the application.

## Future Improvements

- **Database Integration**: Switch from H2 to a more robust database like **PostgreSQL** or **MySQL** for production deployment to handle larger volumes of data.
- **Scalability**: The current setup uses a single instance of the Python API. In a production environment, consider using **Docker** to containerize the Python service and deploy it in a scalable manner using **Kubernetes**.
- **Caching**: Add caching mechanisms using **Redis** or **Ehcache** to reduce the number of redundant calls to the Python API for repeated requests, thereby improving performance.
- **API Security**: Implement **OAuth2** or **JWT** for securing the API endpoints and ensuring that only authorized users can access the prediction features.
- **User Interface**: Develop a simple front-end interface using **React** or **Angular** to allow users to interact with the system more easily, rather than relying on tools like Postman.
- **Improved Model Performance**: Experiment with different machine learning models, such as **Random Forest** or **Neural Networks**, to see if higher accuracy can be achieved. Use **feature importance** analysis to identify which features contribute most to the predictions and consider adding more relevant features.
- **Continuous Integration and Deployment (CI/CD)**: Set up a CI/CD pipeline using tools like **Jenkins** or **GitHub Actions** to automate the process of building, testing, and deploying both the Spring Boot and Python projects.

## Conclusion

This project demonstrates a complete end-to-end application for device price classification, integrating machine learning with a RESTful API interface. The Python model effectively predicts the price range for devices, while the Spring Boot application provides endpoints for CRUD operations and predictions, ensuring a seamless user experience. By adhering to best practices in software development, the project is designed to be modular, maintainable, and easily extensible.

