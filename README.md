# Monkey ECoG

**Reconstruction of the large scale connectome based on macaque ECoG data.**

## Compiler Cython-integrated MI estimator

```bash
cd mutual_information
python setup.py build_ext --inplace
# copy compiled lib (*.so) to the parent folder
cp mutual_info_cy.cpython-37m-darwin.so ..
```