from setuptools import find_packages, setup

package_name = 'welding_pkg'

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
    maintainer='icetenny',
    maintainer_email='ice.tennison@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test = welding_pkg.test_node:main',
            'pcl_to_path = welding_pkg.pcl_to_path:main',
            'pcl_color_filter = welding_pkg.pcl_color_filter:main',
            'welding_path_node = welding_pkg.welding_path_node:main',
            'welding_pose_server = welding_pkg.welding_pose_server:main',
        ],
    },
)
