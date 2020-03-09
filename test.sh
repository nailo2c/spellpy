coverage run -m unittest discover -s tests -p test_*.py
coverage report
coverage xml -o cov.xml
