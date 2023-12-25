import sys
from src.representation.parse_tree import ParseTree
import src.analysis.analysis as analysis
from src.functions.functions import Mathematical
import numpy as np
import src.variation.variation as variation
from decimal import *


def t1():
    n = 10000
    sumd = 0
    sums = 0
    sum_degs = 0
    max_deg = 0
    for i in range(n):
        t = ParseTree()
        t.init_tree(0, 10)
        sumd += t.depth()
        sums += t.size()
        ds, cs = analysis.normalize_polynomial(t)
        sum_degs += ds[-1]
        max_deg = max(ds[-1], max_deg)
    print('av_d', sumd / n, 'av_s', sums / n, 'av_deg', sum_degs / n, 'max_deg', max_deg)


class Poly:
    def __init__(self, ds, cs):
        self.ds = ds
        self.cs = cs

    def additive_op(self, op, poly):
        degrees_left = self.ds
        constants_left = self.cs
        degrees_right = poly.ds
        constants_right = poly.cs
        degrees, constants = [], []
        it_left, it_right = 0, 0
        while it_left < len(degrees_left) and it_right < len(degrees_right):
            if degrees_left[it_left] < degrees_right[it_right]:
                degrees.append(degrees_left[it_left])
                constants.append(constants_left[it_left])
                it_left += 1
            elif degrees_left[it_left] > degrees_right[it_right]:
                degrees.append(degrees_right[it_right])
                constants.append(op(0, constants_right[it_right]))
                it_right += 1
            else:
                degrees.append(degrees_left[it_left])
                constants.append(op(constants_left[it_left], constants_right[it_right]))
                it_left += 1
                it_right += 1
        while it_left < len(degrees_left):
            degrees.append(degrees_left[it_left])
            constants.append(constants_left[it_left])
            it_left += 1
        while it_right < len(degrees_right):
            degrees.append(degrees_right[it_right])
            constants.append(op(0, constants_right[it_right]))
            it_right += 1
        self.ds = degrees
        self.cs = constants
        return self

    def add(self, poly):
        return self.additive_op(Mathematical.add, poly)

    def sub(self, poly):
        return self.additive_op(Mathematical.sub, poly)

    def mul(self, poly):
        degrees_left = self.ds
        constants_left = self.cs
        degrees_right = poly.ds
        constants_right = poly.cs
        degrees, constants = [], []
        ds = np.zeros(len(degrees_left) * len(degrees_right), dtype=int)
        cs = [0] * (len(degrees_left) * len(degrees_right))
        cnt = 0
        for i in range(len(degrees_left)):
            for j in range(len(degrees_right)):
                ds[cnt] = degrees_left[i] + degrees_right[j]
                cs[cnt] = constants_left[i] * constants_right[j]
                cnt += 1
        sorted_ids = np.argsort(ds)
        prv_degree = ds[sorted_ids[0]]
        c = 0
        for i in sorted_ids:
            if prv_degree != ds[i]:
                degrees.append(int(prv_degree))
                constants.append(c)
                c = 0
            c += cs[i]
            prv_degree = ds[i]
        degrees.append(prv_degree)
        constants.append(c)
        self.ds = degrees
        self.cs = constants
        return self

    # integration with limits -1, 1
    def integrate1(self):
        ans, rest = 0, 0
        for d, c in zip(self.ds, self.cs):
            if ~d & 1:
                x = c//(d+1)
                rest += (c-x*(d+1))/(d+1)
                ans += x
        return 2*ans + 2*rest


def test_poly():
    P1 = Poly([2],[1]).add(Poly([1],[1])).mul(Poly([0],[1]))
    print(P1.integrate1())


F = Poly([i for i in range(0, 101)], [(-1)**i * (i + 1) * 10 for i in range(0, 101)])
def loss(y_poly):
    F_copy = Poly(F.ds, F.cs)
    F_copy.sub(y_poly)
    F_copy.mul(F_copy)
    return F_copy.integrate1()


print('Target function:', analysis.sorted_degrees_constants_to_str(F.ds, F.cs), file=sys.stderr)


def mylog(gen, y, new_sub, improve, fy, cs_dist, y_poly):
    print(f'gen:{gen}',  f'depth:{y.depth()}', f'size:{y.size()}', f'new_subspace:{new_sub}', f'improve:{improve}', f'loss:{fy:.3f}', f'cs_dist:{cs_dist:.3f}', analysis.sorted_degrees_constants_to_str(y_poly.ds, y_poly.cs, True), sep=', ', file=sys.stderr)


def euclidean_dist(x_poly, y_poly):
    cs = Poly(x_poly.ds, x_poly.cs).sub(Poly(y_poly.ds, y_poly.cs)).cs
    d = 0
    for c in cs:
        d += c**2
    return Decimal(d).sqrt()


def opoGP(mutation_type):
    MAXGEN = 100000
    x = ParseTree()
    x.symbol = 1
    x_poly = Poly(*analysis.normalize_polynomial(x))
    fx = loss(x_poly)
    print('initial:', fx)

    mylog(0, x, True, True, fx, 1., x_poly)
    for gen in range(1,MAXGEN+1):
        y = x.clone()
        if mutation_type == 'probabilistic_subtree_mutation':
            variation.probabilistic_subtree_mutation(y, 0.1)
        elif mutation_type == 'uniform_subtree_mutation':
            variation.uniform_subtree_mutation(y)
        else:
            raise ValueError(f'No such mutation operator {mutation_type}')
        y_poly = Poly(*analysis.normalize_polynomial(y))
        fy = loss(y_poly)
        new_sub = len(set(d for d,c in zip(y_poly.ds,y_poly.cs) if c!=0)-set(d for d,c in zip(x_poly.ds,x_poly.cs) if c!=0)) != 0
        cs_dist = euclidean_dist(x_poly, y_poly)
        mylog(gen, y, new_sub, fy<fx, fy, cs_dist, y_poly)
        if fy < fx:
            x = y
            fx = fy
            x_poly = Poly(y_poly.ds, y_poly.cs)
            print(f'gen {gen}:', fy)


def parse_poly_str(str_poly):
    ds, cs = [], []
    sign = 1
    for s in str_poly.split('*'):
        for t in s.split(' '):
            if t == '-':
                sign = -1
            elif t == '+':
                sign = 1
            elif t[0] == 'x':
                d = t.split('^')[1]
                ds.append(int(d))
            else:
                cs.append(int(t)*sign)
                sign = 1
    return Poly(ds, cs)


def test_precision():
    str_poly_1 = '-1*x^0 - 105*x^1 + 989*x^2 + 1665*x^3 - 19558*x^4 + 2565*x^5 + 140086*x^6 - 176770*x^7 - 260668*x^8 + 1041271*x^9 - 1265433*x^10 - 1810819*x^11 + 6370468*x^12 - 3201320*x^13 - 5643391*x^14 + 16022525*x^15 - 22721835*x^16 - 3233458*x^17 + 60635536*x^18 - 99381878*x^19 + 5428188*x^20 + 250097061*x^21 - 321701728*x^22 - 197671737*x^23 + 889456751*x^24 - 267363290*x^25 - 1318040369*x^26 + 1137827717*x^27 + 863437875*x^28 - 2466057849*x^29 + 374438199*x^30 + 3603649658*x^31 - 489690416*x^32 - 1922728473*x^33 - 1972958259*x^34 - 4504787189*x^35 + 6168364190*x^36 + 12740541899*x^37 - 12596129866*x^38 - 18148103891*x^39 + 25037749036*x^40 + 20364816382*x^41 - 43859857976*x^42 - 20642247456*x^43 + 60853848676*x^44 + 16716698562*x^45 - 66274341136*x^46 - 6896490088*x^47 + 59205331482*x^48 - 3774275872*x^49 - 46419077762*x^50 + 7902692570*x^51 + 33212443010*x^52 - 4775293946*x^53 - 20571627038*x^54 + 210836352*x^55 + 9423170072*x^56 + 1166349290*x^57 - 2083769070*x^58 + 186042330*x^59 - 634961286*x^60 - 1563413440*x^61 + 541562732*x^62 + 1666599648*x^63 + 90446716*x^64 - 1001473640*x^65 - 305436680*x^66 + 386528240*x^67 + 210934800*x^68 - 92997504*x^69 - 85537200*x^70 + 10219360*x^71 + 22956064*x^72 + 1298432*x^73 - 4062016*x^74 - 679808*x^75 + 432512*x^76 + 104448*x^77 - 20992*x^78 - 6144*x^79'
    str_poly_2 = '10*x^0 - 10*x^1 + 285*x^2 - 2511*x^3 - 19670*x^4 + 16020*x^5 + 267147*x^6 + 122603*x^7 - 2141205*x^8 - 3542075*x^9 + 9938581*x^10 + 28224509*x^11 - 21972356*x^12 - 120179399*x^13 + 3210845*x^14 + 331867630*x^15 - 11660519*x^16 - 509522250*x^17 + 1152735276*x^18 - 929630445*x^19 - 8698874141*x^20 + 9845826967*x^21 + 40334049464*x^22 - 40330409494*x^23 - 145069402755*x^24 + 141218172159*x^25 + 418914452789*x^26 - 511588042549*x^27 - 911852602261*x^28 + 1642413561351*x^29 + 1246374072941*x^30 - 3962518711048*x^31 - 108778003723*x^32 + 6492074208381*x^33 - 4408244595583*x^34 - 5862387858305*x^35 + 12690211446666*x^36 - 851456917338*x^37 - 21101467989844*x^38 + 9276359491323*x^39 + 22795044093940*x^40 - 5678800121767*x^41 - 13052516907666*x^42 - 20090149192582*x^43 - 6085614004628*x^44 + 54606509344534*x^45 + 28313265994682*x^46 - 61595236692730*x^47 - 49592755945946*x^48 + 15820381683396*x^49 + 68208416945212*x^50 + 63360066467196*x^51 - 78624573242400*x^52 - 124490238521420*x^53 + 73234621355100*x^54 + 133250495023196*x^55 - 53588886502780*x^56 - 99716670241472*x^57 + 32309393836972*x^58 + 56971572387836*x^59 - 19679168434028*x^60 - 26935864344452*x^61 + 14464598959716*x^62 + 11563180854336*x^63 - 10447992551464*x^64 - 4620730158384*x^65 + 5576395677812*x^66 + 1489092654080*x^67 - 1710764833528*x^68 - 230719298480*x^69 + 8284485856*x^70 - 97716537480*x^71 + 259648320600*x^72 + 92445492960*x^73 - 128888325952*x^74 - 38348324720*x^75 + 32177265184*x^76 + 9707899488*x^77 - 2999435904*x^78 - 1630943968*x^79 - 860424512*x^80 + 286329536*x^81 + 379502080*x^82 - 72424320*x^83 - 64003968*x^84 + 13191936*x^85 + 5501184*x^86 - 1282560*x^87 - 211968*x^88 + 55296*x^89'
    str_poly_3 = '41*x^0 + 161*x^1 - 676*x^2 - 4720*x^3 - 1103*x^4 + 60723*x^5 - 16081*x^6 - 450667*x^7 + 29615*x^8 + 885682*x^9 + 4414628*x^10 + 11829583*x^11 - 44832131*x^12 - 103808109*x^13 + 198238145*x^14 + 362787159*x^15 - 311498043*x^16 - 430107419*x^17 - 875925482*x^18 - 1848715160*x^19 + 2493869076*x^20 + 14111965498*x^21 + 24466041470*x^22 - 59751252041*x^23 - 188616561916*x^24 + 200951993120*x^25 + 651635507684*x^26 - 592922383755*x^27 - 1390337393765*x^28 + 1548749382975*x^29 + 1965797828864*x^30 - 3320637449647*x^31 - 1772844393846*x^32 + 5227918954025*x^33 + 410143722123*x^34 - 5037849174228*x^35 + 3087897971371*x^36 + 1184744316737*x^37 - 11336476663308*x^38 + 1724600422061*x^39 + 25207267649101*x^40 + 8985152114198*x^41 - 34460074646615*x^42 - 42555778918229*x^43 + 14758703696830*x^44 + 84787364387468*x^45 + 52704283756696*x^46 - 94820040835994*x^47 - 149106273638255*x^48 + 35792524863636*x^49 + 214073152707469*x^50 + 82752012899910*x^51 - 192759691495465*x^52 - 198193415011187*x^53 + 92540917014935*x^54 + 242983187197658*x^55 + 19727889804046*x^56 - 202765341997694*x^57 - 81747030177158*x^58 + 120706313949904*x^59 + 84517789749892*x^60 - 49255373834872*x^61 - 57859685156698*x^62 + 10269686489132*x^63 + 30237294835272*x^64 + 2932883650430*x^65 - 12298662375504*x^66 - 4219856518666*x^67 + 3483721383972*x^68 + 2423270527880*x^69 - 324369897978*x^70 - 881952981604*x^71 - 279085906956*x^72 + 176820871412*x^73 + 175289384088*x^74 + 7621157572*x^75 - 52543067192*x^76 - 18478705240*x^77 + 8345501912*x^78 + 6754262048*x^79 - 206540336*x^80 - 1446187808*x^81 - 205830688*x^82 + 199897696*x^83 + 48149568*x^84 - 17411328*x^85 - 5549568*x^86 + 877440*x^87 + 346368*x^88 - 19968*x^89 - 9216*x^90'
    print(str_poly_3)
    P1 = parse_poly_str(str_poly_3)
    assert str_poly_3 == analysis.sorted_degrees_constants_to_str(P1.ds, P1.cs, True)
    print(loss(P1))


opoGP('uniform_subtree_mutation')
# test_precision()
