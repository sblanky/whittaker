import pandas as pd
import numpy as np
import scipy.constants
from CoolProp.CoolProp import PropsSI
from pygaps.utilities.exceptions import ParameterError


def heat_vap(
    p : float,
    sorptive : str,
):
    r"""
    Determines the enthalpy of vaporsiation, :math:`\lambda_{p}` for a
    specie at a given pressure.

    Parameters
    ----------
    p : float
        Pressure to calculate at
    sorptive : string
        Specie in question

    Returns
    -------
    lambda_p : float
        Enthalpy of vaporisation
    """
    lambda_V = PropsSI('HMOLAR', 'P',
                       p, 'Q', 1,
                       sorptive)
    lambda_L = PropsSI('HMOLAR', 'P',
                       p, 'Q', 0,
                       sorptive)
    lambda_p = lambda_V - lambda_L

    return lambda_p


def whittaker(
    isotherm : "ModelIsotherm",
    p_sat : float = None,
    loading : list = None,
):
    r"""

    Calculate the isosteric heat of adsorption using a single isotherm via the
    Whittaker method.

    Parameters
    ----------
    isotherm : ModelIsotherm
        The model isotherm used. Must be either Toth or Langmuir
    p_sat : float
        The saturation pressure of the sorptive, either 'real' or derived from
        pseudo-saturation
    loading : list[float]
        The loadings for which to calculate the isosteric heat of adsorption

    Returns
    -------
    df : DataFrame
        DataFrame of isosteric heat of adsorption at input loadings.

    Raises
    ------
    ParameterError
        When incorrect type of model isotherm is used.

    Notes
    -----

    The Whittaker method, sometimes known as the Toth potential uses variables
    derived from fitting of a model isotherm (Langmuir or Toth) to derive the
    isosteric enthalpy of adsorption :math:`q_{st}`. The general form of the
    equation is;

    .. math::
        q_{st} = \Delta \lambda + \lambda_{p} + RT

    Where :math:`\Delta \lambda` is the adsorption potential, and
    :math:`\lambda_{p} is the latent heat of the liquid-vapour change at
    equilibrium pressure.

    Whittaker determined :math:`\Delta \lambda` as;

    .. math::
        \Delta \lambda = RT\ln{\left[\left(\frac{p^{sat}}{b^{\frac{1}{t}}\right)\left(\frac{\Theta^{t}}{1-\Theta^{t}}\right) \right]}

    Where :math:`p^{sat}` is the saturation pressure, :math:`\Theta` is the
    fractional coverage, and :math:`b` is derived from the equilbrium constant,
    :math:`K` as :math:`b = \frac{1}{K^t}`. In the case that the adsorptive is
    above is supercrictial, the pseudo saturation pressure is used;
    :math:`p^{sat} = p_c \left(\frac{T}{T_c}\right)^2`. 

    The exponent :math:`t` is only relevant to the Toth version of the method,
    as for the Langmuir model it reduces to 1. Thus, :math:`\Delta \lambda`
    becomes

    .. math::
        \Delta \lambda = RT\ln{\left(\frac{p^{sat}}{b^{\frac{1}{t}}\right)
    """

    if isotherm.model.name not in ['Langmuir', 'Toth']:
        raise ParameterError('''Whittaker method requires either a Langmuir or Toth
                         model isotherm''')
    else:
        R = scipy.constants.R
        n_m = isotherm.model.params['n_m']
        K = isotherm.model.params['K']
        T = isotherm.temperature

        if isotherm.model.name == 'Langmuir':
            t = 1
        else:
            t = isotherm.model.params['t'] # equivalent to m in Whittaker

        b = 1 / (K**t)

        df = pd.DataFrame(columns=['Loading', 'q_st'])

        first_bracket = p_sat / (b**(1/t)) # don't need to calculate every time
        for n in loading:
            p = isotherm.pressure_at(n) * 1000
            sorptive = str(isotherm.adsorbate)
            lambda_p = heat_vap(p, sorptive)
            theta = n / n_m
            theta_t = theta**t
            second_bracket = (theta_t / (1 - theta_t))**((t-1)/t)
            d_lambda = R * T * np.log(first_bracket * second_bracket)
            q_st = d_lambda + lambda_p + (R*T)
            df = df.append(pd.DataFrame({'Loading': [n],
                                         'q_st': [q_st]
                                        })
                          )

        df.reset_index(inplace=True)
        df.drop('index', axis=1, inplace=True)

        return df


if __name__ == '__main__':
    import pygaps.parsing as pgp
    import pygaps.modelling as pgm
    import pygaps
    import glob
    import matplotlib.pyplot as plt

    p_c = 4.5992 * 1000
    T_c = 190.56
    p_sat = p_c * ((298 / T_c)**2)
    model_pressures = np.linspace(10, 10000, 100)
    loading = np.linspace(0.2, 10, 120)
    lambda_p = 8190

    path = './data/'
    data = glob.glob(f'{path}*.aiff')

    fig, axs = plt.subplots(nrows=len(data), ncols=2,
                            figsize=(6, 12),
                            constrained_layout=True,
                            sharex='col')

    for i, d in enumerate(data):
        name = d.split(path)[1][:-5]
        isotherm = pgp.isotherm_from_aif(d)
        isotherm.convert(
            pressure_unit='kPa',
            loading_unit='mol',
            material_unit='kg',
        )

        model_isotherm = pgm.model_iso(
            isotherm,
            branch='ads',
            model='Langmuir',
            verbose=True,
        )

        new_pointisotherm = pygaps.PointIsotherm.from_modelisotherm(
            model_isotherm,
            pressure_points=model_pressures
            )

        pointisotherm_raw = new_pointisotherm.data_raw
        model_info = pd.DataFrame([[model_isotherm.model.name,
                                    model_isotherm.model.rmse]
                                   ]
                                 )
        isotherm_exp = isotherm.data_raw
        isotherm_out = pd.concat([isotherm_exp,
                                  pointisotherm_raw,
                                  model_info
                                  ],
                                 axis=1,
                                 ignore_index=True
                       )

        q_st = whittaker(model_isotherm,
                        p_sat,
                        loading)

        results = pd.concat([isotherm_out, q_st],
                            axis=1,
                            ignore_index=True)

        results.columns = ['exp_pressure', 'exp_loading',
                           'None_1',
                           'model_pressure', 'model_loading',
                           'None_2',
                           'model', 'rmse',
                           'loading', 'q_st',
                            ]

        results.exp_pressure = results.exp_pressure / 100
        results.model_pressure = results.model_pressure / 100
        results.q_st = results.q_st / 1000

        axs[i, 0].annotate(f'{name}\n{round(results.loc[0, "rmse"], 3)}',
                           xy=(0.82, 0.05), xycoords='axes fraction')
        axs[i, 0].scatter(results.exp_pressure, results.exp_loading,
                          clip_on=False,
                          marker='<',
                          fc='none',
                          ec='k')
        axs[i, 0].plot(results.model_pressure, results.model_loading,
                       color='green')
        axs[i, 1].plot(results.loading, results.q_st,
                       color='green')

        results.to_csv(f'./results/langmuir/{name}.csv')

    for l in range(len(data)):
        axs[l, 0].set_xlim(0, 100)
        axs[l, 0].set_ylim(0, axs[l, 0].get_ylim()[1])
        axs[l, 1].set_xlim(0, 10)

        axs[l, 1].yaxis.tick_right()
        axs[l, 1].yaxis.set_label_position('right')

        axs[l, 0].set_ylabel('$\\rm{C_e\ /\ mmol\ g^{-1}}$')
        axs[l, 1].set_ylabel('$\\rm{Q_{st}\ /\ kJ\ mol^{-1}}$')

    axs[len(data)-1, 0].set_xlabel('$\\rm{P\ /\ bar}$')
    axs[len(data)-1, 1].set_xlabel('$\\rm{C_e\ /\ mmol\ g^{-1}}$')

    fig.savefig('./results/langmuir/plot.png',
                bbox_inches='tight', dpi=300)
