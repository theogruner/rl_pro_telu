from setuptools import setup

setup(name='rl_pro_telu',
      version='0.1.0',
      description='MPO and DDPG implementation',
      url='https://github.com/theogruner/rl_pro_telu',
      author='Theo Gruner, Luca Dziarski',
      author_email='theo.gruner@gmail.com, luca@dziarski.de',
      license='GNU',
      packages=['ddpg', 'mpo'],
      install_requires=[
            'gym',
            'numpy',
            'tensorflow',
            'tensorboard',
            'tensorboardX'
      ],
      depenency_links=[
            'https://git.ias.informatik.tu-darmstadt.de/quanser/clients/tree/master'
      ],
      zip_safe=False)
