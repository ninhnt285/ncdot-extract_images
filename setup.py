from setuptools import find_packages, setup

package_name = 'extract_images'

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
    maintainer='tnguy248',
    maintainer_email='tnguy248@uncc.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'extract_images = extract_images.image_extractor:main',
            'get_depth = extract_images.get_depth:main'
        ],
    },
)
