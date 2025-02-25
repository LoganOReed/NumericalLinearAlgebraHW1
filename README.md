## Fortran using F2py
Compiles fortran code as python extension modules which are importable as in the example
Call specific files through the module to ensure fortran modules are linked correctly

## Potential workflow
1. Write fortran subroutine in app
2. Run the following from project directory
```
python -m app.example
```

## TODO 
exclude compiled files from git repo
