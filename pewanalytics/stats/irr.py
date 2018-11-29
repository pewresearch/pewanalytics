import math
import scipy


def kappa_sample_size_power(rate1, rate2, k1, k0, alpha=0.05, power=0.8, twosided=False):

    """
    Translated from N.cohen.kappa: https://cran.r-project.org/web/packages/irr/irr.pdf
    :param rate1: the probability that the first rater will record a positive diagnosis
    :param rate2: the probability that the second rater will record a positive diagnosis
    :param k1: the true Cohen's Kappa statistic
    :param k0: the value of kappa under the null hypothesis
    :param alpha: type I error of test
    :param power: the desired power to detect the difference between true kappa and hypothetical kappa
    :param twosided: TRUE if test is two-sided
    :return: returns required sample size
    """

    if not twosided:
        d = 1
    else:
        d = 2
    pi_rater1 = 1 - rate1
    pi_rater2 = 1 - rate2
    pie = rate1 * rate2 + pi_rater1 * pi_rater2
    pi0 = k1 * (1 - pie) + pie
    pi22 = (pi0 - rate1 + pi_rater2)/2
    pi11 = pi0 - pi22
    pi12 = rate1 - pi11
    pi21 = rate2 - pi11
    pi0h = k0 * (1 - pie) + pie
    pi22h = (pi0h - rate1 + pi_rater2)/2
    pi11h = pi0h - pi22h
    pi12h = rate1 - pi11h
    pi21h = rate2 - pi11h
    Q = (1 - pie)**(-4) * (pi11 * (1 - pie - (rate2 + rate1) *
    (1 - pi0))**2 + pi22 * (1 - pie - (pi_rater2 + pi_rater1) * (1 -
    pi0))**2 + (1 - pi0)**2 * (pi12 * (rate2 + pi_rater1)**2 + pi21 *
    (pi_rater2 + rate1)**2) - (pi0 * pie - 2 * pie + pi0)**2)
    Qh = (1 - pie)**(-4) * (pi11h * (1 - pie - (rate2 + rate1) *
    (1 - pi0h))**2 + pi22h * (1 - pie - (pi_rater2 + pi_rater1) *
    (1 - pi0h))**2 + (1 - pi0h)**2 * (pi12h * (rate2 + pi_rater1)**2 +
    pi21h * (pi_rater2 + rate1)**2) - (pi0h * pie - 2 * pie +
    pi0h)**2)
    N = ((scipy.stats.norm.ppf(1 - alpha/d) * math.sqrt(Qh) + scipy.stats.norm.ppf(power) * math.sqrt(Q))/(k1 - k0))**2
    return math.ceil(N)


def kappa_sample_size_CI(kappa0, kappaL, props, kappaU=None, alpha=0.05):

    """
    Translated from kappaSize: https://github.com/cran/kappaSize/blob/master/R/CIBinary.R
    :param kappa0: The preliminary value of kappa
    :param kappaL: The desired expected lower bound for a two-sided 100(1 - alpha) % confidence interval for kappa. Alternatively, if kappaU is set to NA, the procedure produces the number of required subjects for a one-sided confidence interval
    :param props: The anticipated prevalence of the desired trait
    :param kappaU: The desired expected upper confidence limit for kappa
    :param alpha: The desired type I error rate
    :return:
    """

    if not kappaU:
        chiCrit = scipy.stats.chi2.ppf((1 - 2 * alpha), 1)

    if kappaL and kappaU:
        chiCrit = scipy.stats.chi2.ppf((1 - alpha), 1)

    def CalcIT(rho0, rho1, Pi, n):
        def P0(r, p):
            x = (1 - p) ** 2 + r * p * (1 - p)
            return (x)

        def P1(r, p):
            x = 2 * (1 - r) * p * (1 - p)
            return (x)

        def P2(r, p):
            x = p ** 2 + r * p * (1 - p)
            return (x)

        r1 = ((n * P0(r=rho0, p=Pi)) - (n * P0(r=rho1, p=Pi))) ** 2 / (n * P0(r=rho1, p=Pi))
        r2 = ((n * P1(r=rho0, p=Pi)) - (n * P1(r=rho1, p=Pi))) ** 2 / (n * P1(r=rho1, p=Pi))
        r3 = ((n * P2(r=rho0, p=Pi)) - (n * P2(r=rho1, p=Pi))) ** 2 / (n * P2(r=rho1, p=Pi))
        return sum([r for r in [r1, r2, r3] if r])

    n = 10
    result = 0
    while (result - .001) < chiCrit:
        result = CalcIT(kappa0, kappaL, props, n)
        n += 1
    return n