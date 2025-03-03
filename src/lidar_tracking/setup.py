from setuptools import find_packages, setup
import os
import glob

package_name = 'lidar_tracking'
python_scripts = [
    os.path.splitext(os.path.basename(script))[0] + " = " + package_name + "." + os.path.splitext(os.path.basename(script))[0] + ":main"
    for script in glob.glob(os.path.join(package_name, "*.py"))
    if not script.endswith("__init__.py") 
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='addinedu',
    maintainer_email='fjdk78945@gamil.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': python_scripts,
    },
)
