

pip uninstall -y pytorch3dunet
if %ERRORLEVEL% neq 0 (
    echo Failed to uninstall pytorch3dunet
    exit /b %ERRORLEVEL%
)

cd ..
if %ERRORLEVEL% neq 0 (
    echo Failed to change directory to parent
    exit /b %ERRORLEVEL%
)

call python setup.py install
if %ERRORLEVEL% neq 0 (
    echo Failed to install pytorch3dunet
    exit /b %ERRORLEVEL%
)

cd Data_tool
if %ERRORLEVEL% neq 0 (
    echo Failed to change directory to Date_tool
    exit /b %ERRORLEVEL%
)

call streamlit run app.py
if %ERRORLEVEL% neq 0 (
    echo Failed to run Streamlit app
    exit /b %ERRORLEVEL%
)

echo All commands executed successfully
