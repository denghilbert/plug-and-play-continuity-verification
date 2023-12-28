import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

data = [(0.5, 0, 0.6381588258605072), (0.5, 10, 0.6235068500905797), (0.5, 20, 0.6140349014945652), (0.5, 30, 0.6132741734601449), (0.5, 40, 0.6127363564311594), (0.5, 50, 0.6118128679800725), (1.5, 0, 0.740920799365942), (1.5, 10, 0.7228897758152174), (1.5, 20, 0.6721439085144928), (1.5, 30, 0.6472238734148551), (1.5, 40, 0.6333255491394928), (1.5, 50, 0.6282728996829711), (2.0, 0, 0.7578974184782609), (2.0, 10, 0.7452799479166666), (2.0, 20, 0.6994664288949275), (2.0, 30, 0.6664048346920289), (2.0, 40, 0.6479421422101449), (2.0, 50, 0.6405400815217391), (2.5, 0, 0.7734445765398551), (2.5, 10, 0.7551375679347826), (2.5, 20, 0.7217221467391305), (2.5, 30, 0.681640625), (2.5, 40, 0.6621942934782609), (2.5, 50, 0.6523119055706522), (3.0, 0, 0.7774286684782609), (3.0, 10, 0.7599496150362319), (3.0, 20, 0.742477638134058), (3.0, 30, 0.7004429913949275), (3.0, 40, 0.6796167346014492), (3.0, 50, 0.6653610450634058), (3.5, 0, 0.7797356204710145), (3.5, 10, 0.7648748867753623), (3.5, 20, 0.7553852468297102), (3.5, 30, 0.7127915534420289), (3.5, 40, 0.6918662250905797), (3.5, 50, 0.6773168591485508), (4.0, 0, 0.7823539402173914), (4.0, 10, 0.7684131567028986), (4.0, 20, 0.7615984488224637), (4.0, 30, 0.726364356884058), (4.0, 40, 0.7035885133605072), (4.0, 50, 0.6878679800724637), (4.5, 0, 0.7858143682065217), (4.5, 10, 0.7733879642210145), (4.5, 20, 0.769729393115942), (4.5, 30, 0.7330870697463768), (4.5, 40, 0.7137044270833334), (4.5, 50, 0.6967879585597826), (5.0, 0, 0.7879585597826086), (5.0, 10, 0.7792544157608695), (5.0, 20, 0.7724609375), (5.0, 30, 0.7389110620471014), (5.0, 40, 0.7213223222373188), (5.0, 50, 0.7051630434782609), (5.5, 0, 0.7887369791666666), (5.5, 10, 0.7809810914855072), (5.5, 20, 0.772064651268116), (5.5, 30, 0.7463272758152174), (5.5, 40, 0.7273055366847826), (5.5, 50, 0.7120697463768116), (6.0, 0, 0.7908740942028986), (6.0, 10, 0.7853402400362319), (6.0, 20, 0.775192481884058), (6.0, 30, 0.7517337522644928), (6.0, 40, 0.7325492527173914), (6.0, 50, 0.7172002377717391), (6.5, 0, 0.7910439311594203), (6.5, 10, 0.7871447576992754), (6.5, 20, 0.7816816689311594), (6.5, 30, 0.7582017096920289), (6.5, 40, 0.7388119904891305), (6.5, 50, 0.7227907042572463), (7.0, 0, 0.789947067481884), (7.0, 10, 0.7850288722826086), (7.0, 20, 0.7811650815217391), (7.0, 30, 0.7629712975543478), (7.0, 40, 0.7429729959239131), (7.0, 50, 0.7276593636775363), (7.5, 0, 0.7909307065217391), (7.5, 10, 0.785736526268116), (7.5, 20, 0.7851420969202898), (7.5, 30, 0.7655825407608695), (7.5, 40, 0.7474736752717391), (7.5, 50, 0.73193359375), (8.0, 0, 0.7906830276268116), (8.0, 10, 0.7874207427536232), (8.0, 20, 0.7859912817028986), (8.0, 30, 0.7682857789855072), (8.0, 40, 0.7515992980072463), (8.0, 50, 0.7363776607789855), (8.5, 0, 0.7905061141304348), (8.5, 10, 0.7882486979166666), (8.5, 20, 0.7863804913949275), (8.5, 30, 0.7716117527173914), (8.5, 40, 0.753998245018116), (8.5, 50, 0.7400857676630435), (9.0, 0, 0.7904141191123188), (9.0, 10, 0.7900815217391305), (9.0, 20, 0.7904990375905797), (9.0, 30, 0.7739328577898551), (9.0, 40, 0.7579611073369565), (9.0, 50, 0.745223335597826), (9.5, 0, 0.7906901041666666), (9.5, 10, 0.7903079710144928), (9.5, 20, 0.7902230525362319), (9.5, 30, 0.7737488677536232), (9.5, 40, 0.7613507699275363), (9.5, 50, 0.7486200747282609), (10.0, 0, 0.7932871942934783), (10.0, 10, 0.7904141191123188), (10.0, 20, 0.7926149230072463), (10.0, 30, 0.7763105751811594), (10.0, 40, 0.7640610846920289), (10.0, 50, 0.7537717957427537), (10.5, 0, 0.7923177083333334), (10.5, 10, 0.7911642323369565), (10.5, 20, 0.7948086503623188), (10.5, 30, 0.7801460597826086), (10.5, 40, 0.7646767436594203), (10.5, 50, 0.7552507925724637), (11.0, 0, 0.7925936933876812), (11.0, 10, 0.7912562273550725), (11.0, 20, 0.7961461163949275), (11.0, 30, 0.7807334125905797), (11.0, 40, 0.7662265058876812), (11.0, 50, 0.7568430140398551), (11.5, 0, 0.7914472939311594), (11.5, 10, 0.7919497282608695), (11.5, 20, 0.7959975090579711), (11.5, 30, 0.7808395606884058), (11.5, 40, 0.7670615375905797), (11.5, 50, 0.758682914402174), (12.0, 0, 0.7914826766304348), (12.0, 10, 0.7926856884057971), (12.0, 20, 0.7954455389492754), (12.0, 30, 0.781158004981884), (12.0, 40, 0.7692835711050725), (12.0, 50, 0.7595957880434783), (12.5, 0, 0.7909307065217391), (12.5, 10, 0.7915534420289855), (12.5, 20, 0.7957569067028986), (12.5, 30, 0.7815967504528986), (12.5, 40, 0.770705955615942), (12.5, 50, 0.7612516983695652), (13.0, 0, 0.7893172554347826), (13.0, 10, 0.792084182518116), (13.0, 20, 0.7948086503623188), (13.0, 30, 0.7830474411231884), (13.0, 40, 0.7699346127717391), (13.0, 50, 0.7628226902173914), (13.5, 0, 0.7879302536231884), (13.5, 10, 0.7922894021739131), (13.5, 20, 0.795013870018116), (13.5, 30, 0.7824883944746377), (13.5, 40, 0.7719018908514492), (13.5, 50, 0.7638912477355072)]
# Assuming x, y, z are evenly spaced and form a grid
#x = np.linspace(-5, 5, 100)
#y = np.linspace(-5, 5, 100)
#x, y = np.meshgrid(x, y)
#z = np.sin(np.sqrt(x**2 + y**2))


x, y, z = zip(*data)

# Creating a new figure
fig = plt.figure()

# Adding a subplot with 3D projection
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x, y, z)

# Adding labels
ax.set_xlabel('classifier-free scale')
ax.set_ylabel('timesteps_control')
ax.set_zlabel('CLIP-text_image-semantic')

# Display the plot
plt.show()

