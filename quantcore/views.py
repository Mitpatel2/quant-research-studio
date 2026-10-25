"""Research interfaces. Data source and model assumptions stay visible."""
import io
import json
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from .options import black_scholes,price,implied_vol,demo_chain,clean_chain,calibrate,predict_surface,surface_diagnostics,carry_at,surface_strikes,surface_maturities
from .data import cached_futures,load_candles,ROOT
from .regimes import walk_forward
from .events import Risk,run


def theme():
    st.set_page_config(page_title='Quant Research Studio',page_icon='◈',layout='wide')
    st.markdown('''<style>
    .block-container {max-width:1450px;padding-top:4rem}
    [data-testid="stMetric"] {background:#141d2d;border:1px solid #27384e;padding:15px;border-radius:10px}
    .eyebrow {letter-spacing:.18em;color:#91adc9;font-size:12px;margin-bottom:8px}
    </style>''',unsafe_allow_html=True)


def options_view():
    st.markdown('<div class="eyebrow">01 / DERIVATIVES LAB</div>',unsafe_allow_html=True)
    st.title('Options, made visible')
    st.write('Black–Scholes pricing · implied volatility · gradient descent · risk sensitivities')
    st.caption('European options • annual continuously compounded rates • ACT/365 • long-option, per-unit values')
    calc,surface,data_tab=st.tabs(['Calculator & formulas','3D surface & Greeks','Data & assumptions'])
    with calc:
        cols=st.columns(3)
        s=cols[0].number_input('Underlying price S',min_value=.01,value=100.,step=1.,key='bs_s')
        k=cols[1].number_input('Strike K',min_value=.01,value=100.,step=1.,key='bs_k')
        days=cols[2].number_input('Days to expiry',min_value=0.,max_value=3650.,value=365.,step=1.,key='bs_days')
        cols=st.columns(3)
        vol=cols[0].number_input('Volatility σ (%)',min_value=0.,max_value=500.,value=20.,step=1.,key='bs_vol')/100
        rate=cols[1].number_input('Risk-free rate r (%)',min_value=-10.,max_value=50.,value=5.,step=.25,key='bs_r')/100
        div=cols[2].number_input('Dividend yield q (%)',min_value=0.,max_value=50.,value=0.,step=.25,key='bs_q')/100
        kind=st.radio('Option type',['call','put'],horizontal=True,key='bs_kind')
        result=black_scholes(s,k,days/365,rate,vol,div,kind)
        cols=st.columns(3)
        cols[0].metric(f'{kind.title()} value',f"{result['price']:.6f}")
        cols[1].metric('d₁','Undefined at boundary' if result['d1'] is None else f"{result['d1']:.6f}")
        cols[2].metric('d₂','Undefined at boundary' if result['d2'] is None else f"{result['d2']:.6f}")
        st.subheader('The five Greeks')
        cols=st.columns(5)
        units={'delta':'per +1 underlying unit','gamma':'delta per +1 underlying unit','vega':'per +1 percentage point IV','theta':'per calendar day elapsed','rho':'per +1 percentage point rate'}
        for col,name in zip(cols,units):
            col.metric(name.title(),'—' if result[name] is None else f'{result[name]:.6f}')
            col.caption(units[name])
        if result['d1'] is None:
            st.info('At expiry or zero volatility, price uses the intrinsic/discounted deterministic limit. Smooth Greeks and d₁/d₂ are not reported at these boundaries.')
        with st.expander('Black–Scholes formulas and variables',expanded=True):
            st.latex(r'C=S e^{-qT}N(d_1)-K e^{-rT}N(d_2)')
            st.latex(r'P=K e^{-rT}N(-d_2)-S e^{-qT}N(-d_1)')
            st.latex(r'd_1=\frac{\ln(S/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt{T}}')
            st.latex(r'd_2=d_1-\sigma\sqrt{T}')
            st.write('Set q = 0 to obtain the exact call formula in your screenshots. T = days / 365; σ is annualized volatility; N is the standard normal cumulative distribution; e is the exponential base.')
            st.write('S: current underlying price · K: strike · r: annual risk-free rate · q: continuous dividend yield. Prices are per underlying unit; multiply by the actual contract multiplier for contract dollars.')
            st.caption('Greeks are model partial derivatives holding volatility fixed; they are not a total surface-recalibration sensitivity or a prediction of realized P&L.')
        with st.expander('Recover implied volatility from a price'):
            observed=st.number_input('Observed option price',min_value=0.,value=float(result['price']),step=.1,key='bs_observed')
            try:st.write(f'Implied volatility: {100*implied_vol(observed,s,k,days/365,rate,div,kind):.6f}%')
            except ValueError as err:st.warning(str(err))
        st.download_button('Export calculation',json.dumps(dict(inputs=dict(S=s,K=k,days=days,rate=rate,volatility=vol,dividend_yield=div,kind=kind),results=result,units=units),indent=2),file_name='black-scholes.json',mime='application/json',key='calc_export')
    with surface:
        cached_snapshot=ROOT/'data/databento/2026-10-23-spxw/snapshot-1435.csv'
        dense_snapshot=ROOT/'data/databento/2026-10-23-spxw/snapshot-dense-1435.csv'
        if dense_snapshot.exists():cached_snapshot=dense_snapshot
        choices=['Analytic demo (generated)','Import snapshot CSV']
        if cached_snapshot.exists():choices.append('Cached Databento SPXW')
        source=st.radio('Quote source',choices,horizontal=True,key='quote_source')
        raw=None
        if source.startswith('Analytic'):
            st.warning('Generated analytic quotes for demonstration. These points are not observed market data.')
            raw=demo_chain()
        elif source=='Cached Databento SPXW':
            raw=pd.read_csv(cached_snapshot)
            st.info(f'Real historical Databento SPXW quotes · October 23, 2026, 10:35 a.m. Eastern · {raw.expiry.nunique()} observed expiries · matching SPX index observation. This is a cached research snapshot.')
            st.caption('CBBO records mark minute boundaries; actual last bid/ask update times are unavailable. Cboe verifies European exercise, PM settlement and the $100 multiplier.')
        else:
            st.caption('Imported CSV provenance is supplied by its owner and is not independently authenticated.')
            uploaded=st.file_uploader('Options snapshot CSV',type=['csv'],key='chain_upload')
            if uploaded is None:st.info('Upload a European options snapshot with independently checked provenance and matching underlying inputs.')
            else:
                try:raw=pd.read_csv(uploaded)
                except Exception as err:st.error(f'Unable to read CSV: {err}')
        st.download_button('Download snapshot template',demo_chain().to_csv(index=False),file_name='options-template-generated.csv',mime='text/csv')
        result_slot=st.empty()
        result_slot.empty()  # Clear prior-source results before calibration begins.
        with result_slot.container():
            if raw is not None:
                try:
                    chain,rejections=clean_chain(raw)
                    if len(chain):
                        first=chain.iloc[0]
                        st.caption(f"Snapshot {first['asof']} · spot {first.spot:g} · expiry rates {chain.rate.min()*100:.3f}–{chain.rate.max()*100:.3f}% · expiry carry yields {chain.dividend_yield.min()*100:.3f}–{chain.dividend_yield.max()*100:.3f}%. Surface inputs come from this snapshot, independently of the calculator.")
                        if 'carry_source' in chain and chain.carry_source.eq('training_pairs_put_call_parity').all():
                            st.info('Rates and carry yields were estimated from training call/put pairs at this snapshot. They are model assumptions, not independent rate or dividend observations. Greek sensitivities hold these inputs fixed.')
                    method_label=st.radio('Calibration model',['Corrected expiry smiles','Legacy polynomial'],horizontal=True,key='surface_method')
                    method='corrected' if method_label.startswith('Corrected') else 'legacy'
                    a,b=st.columns(2)
                    iterations=a.slider('Gradient descent iterations',min_value=50,max_value=2000,value=600,step=50)
                    lr=b.select_slider('Learning rate',options=[.001,.003,.01,.03,.1],value=.03)
                    model=calibrate(chain,iterations,lr,method=method)
                    cols=st.columns(3)
                    cols[0].metric('Accepted quotes',len(chain));cols[1].metric('Rejected quotes',len(rejections))
                    cols[2].metric('Fit RMSE (vol points)',f"{model['rmse_iv']*100:.4f}")
                    fitted_prices=np.array([price(row.spot,row.strike,row.years,row.rate,float(v),row.dividend_yield,row.kind) for (_,row),v in zip(chain.iterrows(),model['fitted_iv'])])
                    in_spread=(fitted_prices>=chain.bid.to_numpy())&(fitted_prices<=chain.ask.to_numpy())
                    a,b=st.columns(2)
                    a.metric('Current fit price RMSE (points)',f'{np.sqrt(np.mean((fitted_prices-chain.mid.to_numpy())**2)):.4f}')
                    b.metric('Current fit inside bid/ask',f'{in_spread.mean():.1%}')
                    if source=='Cached Databento SPXW':
                        prefix='dense' if cached_snapshot==dense_snapshot else 'cross-sectional'
                        report_path=ROOT/f'reports/surface-fit-retest/{prefix}-1435-{method}.json'
                        if report_path.exists():
                            saved=json.loads(report_path.read_text());held=saved['metrics']['holdout']
                            message=f"Recorded {method} holdout test: {held['inside_bid_ask_fraction']:.1%} of prices inside bid/ask; price RMSE {held['price_rmse']:.4f} SPX points across {held['quotes']} held-out quote records."
                            if method=='corrected':st.info(message)
                            else:st.warning(message)
                            st.caption('Recorded tests use training-only quotes and a 2,000-step cap. The current chart fits all accepted quotes under the controls above. This reused ten-minute sample does not validate trading performance.')
                            if held.get('unpriced_quotes',0):st.caption(f"{held['unpriced_quotes']} held-out quotes were outside the calibrated strike support. They count as failures in bid/ask coverage and are excluded from pricing RMSE.")
                        expiry_report=ROOT/'reports/surface-fit-retest/expiry-coverage.json'
                        if method=='corrected' and expiry_report.exists() and not json.loads(expiry_report.read_text())['interpolation_validated']:
                            st.warning('Missing-expiry tests still showed pricing errors. The chart uses 22 observed expiry anchors; prices between them remain model estimates and are not validated for trading.')
                    measure=st.selectbox('Surface measure',['Implied volatility','Price','Delta','Gamma','Vega','Theta','Rho'])
                    option=st.radio('Surface option type',['call','put'],horizontal=True,key='surface_kind')
                    rendering=st.radio('Chart view',['3D surface','Heatmap'],horizontal=True)
                    strikes=surface_strikes(model,chain)
                    times=surface_maturities(chain)
                    xx,tt=np.meshgrid(strikes,times);iv=predict_surface(model,xx,tt)
                    rates,yields=carry_at(chain,tt)
                    if measure=='Implied volatility':zz=iv*100;ztitle='Annual IV (%)'
                    else:
                        name=measure.lower();first=chain.iloc[0]
                        zz=np.array([[black_scholes(model['spot'],float(kk),float(t),float(rr),float(vv),float(qq),option)[name] for kk,t,vv,rr,qq in zip(kr,tr,vr,rrs,qqs)] for kr,tr,vr,rrs,qqs in zip(xx,tt,iv,rates,yields)])
                        ztitle=measure+({'Vega':' / 1 vol point','Theta':' / calendar day','Rho':' / 1 rate point'}.get(measure,''))
                    fig=go.Figure()
                    if rendering=='3D surface':
                        fig.add_trace(go.Surface(x=xx,y=tt*365,z=zz,colorscale='Viridis',colorbar_title=ztitle,name='Fitted model'))
                        if measure=='Implied volatility':
                            points=chain[(chain.kind==option)&chain.strike.between(strikes.min(),strikes.max())]
                            fig.add_trace(go.Scatter3d(x=points.strike,y=points.years*365,z=points.iv*100,mode='markers',marker=dict(size=3,color='#ffbd69'),name='Quote-derived points'))
                        # Price, day and Greek units differ by orders of magnitude.
                        # Keep a usable box rather than scaling axes in raw units.
                        fig.update_layout(scene=dict(xaxis_title='Strike',yaxis_title='Days to expiry',zaxis_title=ztitle,
                            aspectmode='cube',camera=dict(eye=dict(x=1.5,y=1.5,z=1.1))),height=550)
                    else:
                        fig.add_trace(go.Heatmap(x=strikes,y=times*365,z=zz,colorscale='Viridis',colorbar_title=ztitle))
                        fig.update_layout(xaxis_title='Strike',yaxis_title='Days to expiry',height=500)
                    fig.update_layout(margin=dict(l=10,r=10,t=20,b=10))
                    st.plotly_chart(fig,width='stretch',key='surface_plot')
                    if method=='corrected':st.caption(f'Displayed strike support: {strikes.min():.2f}–{strikes.max():.2f}, shared by every observed expiry. Quote diagnostics and current-fit metrics include all accepted strikes.')
                    st.caption(('Corrected fit uses price-sensitive per-expiry log-IV smiles and total-variance interpolation within common observed strike support. ' if method=='corrected' else 'Legacy fit is a five-parameter log-volatility polynomial. ')+'Neither is guaranteed globally arbitrage-free. Greek surfaces hold fitted local volatility fixed when differentiating the pricing model.')
                    loss=px.line(x=np.arange(len(model['loss'])),y=model['loss'],labels={'x':'Gradient step','y':'Weighted loss + ridge'})
                    loss.update_layout(height=230,margin=dict(l=10,r=10,t=10,b=10))
                    st.plotly_chart(loss,width='stretch',key='loss_plot')
                    st.caption(f"Completed {model['iterations']} steps · gradient norm {model['gradient_norm']:.3g} · tolerance reached: {model['converged']}")
                    if method=='corrected':st.caption('SVD-preconditioned gradient descent · degree 4 per expiry · ridge 0.0001 · reported gradient norm is in preconditioned coordinates')
                    if not model['converged']:st.warning('This fit reached its step limit without meeting the gradient tolerance. Increase steps or learning rate before interpreting the current surface.')
                    issues=surface_diagnostics(model,chain)
                    if issues:st.warning('Grid diagnostics: '+'; '.join(issues))
                    else:st.info('No violations found on the tested strike/maturity grid. This is not proof over the continuous surface.')
                    fit_frame=chain.assign(fitted_iv=model['fitted_iv'],residual_iv=chain.iv-model['fitted_iv'])
                    st.download_button('Export fitted quotes',fit_frame.to_csv(index=False),file_name='fitted-quotes.csv',mime='text/csv',on_click='ignore')
                    st.download_button(f'Export {method} {option} {measure.lower()} grid',pd.DataFrame(dict(strike=xx.ravel(),days=tt.ravel()*365,value=zz.ravel(),measure=measure,kind=option,calibration_method=method,asof=str(chain['asof'].iloc[0]),spot=model['spot'],rate=rates.ravel(),dividend_yield=yields.ravel(),fitted_iv=iv.ravel())).to_csv(index=False),file_name=f'{method}-{measure.lower().replace(" ","-")}-{option}-surface-grid.csv',mime='text/csv',on_click='ignore',key=f'grid_export_{method}_{measure}_{option}')
                    with st.expander('Quote diagnostics'):
                        st.dataframe(fit_frame[['strike','years','kind','bid','ask','rate','dividend_yield','iv','fitted_iv','residual_iv']],width='stretch')
                        if len(rejections):st.dataframe(rejections,width='stretch')
                except (ValueError,TypeError,KeyError) as err:st.error(str(err))
    with data_tab:
        st.subheader('Data that you can trace')
        st.write('Real quote import requires asof, quote_time, expiry, spot, strike, bid, ask, rate, dividend_yield, kind and exercise_style. Timestamps must include their timezone. One as-of snapshot and spot; one rate and carry yield per expiry. Snapshot-record age is at most 60 seconds. Interval BBO records can contain carried-forward quotes and do not establish the last bid/ask update time. American options are rejected.')
        st.write('This public copy supports generated analytic quotes and licensed CSV imports. The private download/cache workflow is not included. No API key is collected by this UI.')
        st.markdown('[Black–Scholes reference](https://www.mathworks.com/help/finance/blsprice.html) · [OIC Greeks](https://www.optionseducation.org/advancedconcepts/understanding-options-greeks)')


def candle_source(prefix):
    source=st.radio('Candle source',['Existing local NQ history','Import M15 CSV'],horizontal=True,key=prefix+'_source')
    try:
        if source.startswith('Existing'):
            frame=cached_futures()
            st.caption('Local NQ price history, June 2025–June 2026. Used as an MNQ proxy; previously inspected, not an untouched holdout.')
        else:
            f=st.file_uploader('M15 OHLCV CSV',type=['csv'],key=prefix+'_upload')
            if f is None:st.info('Choose a file with timezone-aware bar-open timestamps.');return None
            frame=load_candles(f)
        st.caption(f'{len(frame):,} candles · {frame.index.min()} → {frame.index.max()} · gaps remain gaps')
        return frame
    except (ValueError,TypeError,KeyError) as err:st.error(str(err));return None


def regimes_view():
    st.markdown('<div class="eyebrow">02 / MARKET STRUCTURE</div>',unsafe_allow_html=True)
    st.title('Market regime atlas')
    st.write('Three descriptive clusters, trained on past observations. No future-fit labels.')
    st.caption('Futures sessions use the Chicago 17:00 opening boundary. Features are available only after each M15 candle closes.')
    frame=candle_source('regime')
    with st.form('regime_windows'):
        a,b=st.columns(2)
        train=a.number_input('Training sessions',min_value=20,max_value=180,value=60,step=10)
        test=b.number_input('Test sessions per fold',min_value=5,max_value=60,value=40,step=5)
        submitted=st.form_submit_button('Run regime analysis',disabled=frame is None)
        st.caption('Submit changes to apply them. Results and exports below describe the last completed run.')
    if frame is None:return
    try:
        source_hash=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
        if submitted or st.session_state.get('regime_data_hash')!=source_hash or 'regime_result' not in st.session_state:
            st.session_state.pop('regime_result',None)
            rows,folds=walk_forward(frame,int(train),int(test))
            st.session_state.regime_result=(rows,folds,int(train),int(test))
            st.session_state.regime_data_hash=source_hash
        rows,folds,completed_train,completed_test=st.session_state.regime_result
        st.info(f'Completed run: {completed_train} training sessions; {completed_test} test sessions per fold. Exports use these settings.')
        labels=['Lower volatility','Middle volatility','Higher volatility']
        colors={'Lower volatility':'#66c2a5','Middle volatility':'#fcbd62','Higher volatility':'#d77991'}
        fig=px.scatter(rows.reset_index(),x=rows.index.name or 'index',y='close',color='regime',color_discrete_map=colors,hover_data=['fold','training_end'])
        fig.update_traces(marker_size=3);fig.update_layout(height=430,xaxis_title='UTC candle open',yaxis_title='NQ price')
        st.plotly_chart(fig,width='stretch')
        st.caption('Labels are ordered by training-cluster volatility. They do not mean buy/sell, bull/bear, or a probability of profit.')
        st.dataframe(rows.groupby('regime').agg(observations=('close','count'),mean_next_return=('next_return','mean'),next_return_std=('next_return','std')),width='stretch')
        st.caption('Next-bar returns are outcome diagnostics only. They never enter feature scaling, clustering, or current labels. No return is scored across a source gap or fold boundary.')
        st.subheader('Chronological fold record');st.dataframe(pd.DataFrame(folds)[['fold','training_end','test_start','test_end','training_rows','test_rows','inertia']],width='stretch')
        st.download_button('Export regime observations',rows.to_csv(),file_name=f'regimes-train{completed_train}-test{completed_test}.csv',mime='text/csv')
        st.download_button('Export regime model record',json.dumps(dict(training_sessions=completed_train,test_sessions=completed_test,dataset_fingerprint_sha256=source_hash,folds=folds),indent=2),file_name=f'regime-models-train{completed_train}-test{completed_test}.json',mime='application/json')
    except ValueError as err:st.error(str(err))


def events_view():
    st.markdown('<div class="eyebrow">03 / EXECUTION RESEARCH</div>',unsafe_allow_html=True)
    st.title('Follow every event')
    st.write('Completed candle → signal → next-open order → risk review → fill → exit')
    frame=candle_source('events')
    st.caption('Benchmark: EMA10/30 cross; 1.5×ATR14 stop; 2R target; one attempt per Chicago session. This is an engineering example, not extracted reel logic.')
    a,b,c=st.columns(3)
    cap=a.number_input('Maximum planned loss / trade ($)',min_value=1.,max_value=1000.,value=50.,step=5.)
    daily=b.number_input('Daily realized loss budget ($)',min_value=1.,max_value=5000.,value=100.,step=10.)
    dd=c.number_input('Realized drawdown budget ($)',min_value=1.,max_value=10000.,value=500.,step=50.)
    a,b,c=st.columns(3)
    fee=a.number_input('Round-trip commission / contract ($)',min_value=0.,max_value=20.,value=1.54,step=.1)
    slip=b.number_input('Adverse slippage (ticks)',min_value=0.,max_value=20.,value=1.,step=1.)
    quantity=c.number_input('Contracts',min_value=1,max_value=10,value=1,step=1)
    if frame is None:return
    signature=(hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest(),cap,daily,dd,fee,slip,quantity)
    # Any changed input clears the old report; no stale metrics beside new settings.
    if st.session_state.get('event_signature')!=signature:
        st.session_state.pop('event_result',None)
    if st.button('Run historical comparison',type='primary'):
        result=run(frame,Risk(max_trade_loss=cap,daily_loss=daily,drawdown=dd,commission_rt=fee,slippage_ticks=slip,quantity=int(quantity)))
        stress=run(frame,Risk(max_trade_loss=cap,daily_loss=daily,drawdown=dd,commission_rt=fee*2,slippage_ticks=max(4,slip*2),quantity=int(quantity)))
        st.session_state.event_result=(result,stress);st.session_state.event_signature=signature
    if 'event_result' not in st.session_state:
        st.info('Run the comparison to see results for the current settings.');return
    result,stress=st.session_state.event_result
    if not result['trades']:
        rejected=sum(e['event']=='reject' for e in result['events'])
        st.warning(f'No trades filled under these limits. {rejected} proposed orders were rejected. This is a risk-gate outcome, not a profitability estimate.')
    a,b,c=st.columns(3)
    a.metric('Base net P&L',f"${result['metrics']['net']:,.2f}")
    b.metric('Stress net P&L',f"${stress['metrics']['net']:,.2f}")
    c.metric('Closed trades',result['metrics']['trades'])
    fig=px.line(pd.DataFrame(result['equity']),x='time',y='equity');fig.update_layout(height=330,yaxis_title='Realized equity ($)')
    st.plotly_chart(fig,width='stretch')
    st.caption('Research comparison on reused history; realized equity only. Stress doubles fees and uses at least four ticks slippage. This is not a continuous broker mark-to-market risk engine.')
    st.dataframe(pd.DataFrame([dict(costs='base',**result['metrics']),dict(costs='stress',**stress['metrics'])]),width='stretch')
    trades_tab,events_tab=st.tabs(['Trades','Event log'])
    with trades_tab:st.dataframe(pd.DataFrame(result['trades']),width='stretch')
    with events_tab:
        include_market=st.checkbox('Include source market events',value=False)
        shown=result['events'] if include_market else [e for e in result['events'] if not e['event'].startswith('market_')]
        st.caption(f"Showing {len(shown):,} of {len(result['events']):,} events. The download always includes the full queue trace.")
        st.dataframe(pd.DataFrame(shown),width='stretch')
    st.download_button('Export complete event run',json.dumps(dict(base=result,stress=stress),indent=2,default=str),file_name='event-comparison.json',mime='application/json')


def launch(project=None):
    theme()
    st.sidebar.title('◈ Quant Studio')
    st.sidebar.caption('Three independent research projects')
    names=['Options surface','Market regimes','Event backtester']
    selected=project or st.sidebar.radio('Project',names,key='project_select')
    st.sidebar.divider();st.sidebar.write('Local research • no broker orders')
    st.sidebar.caption('Generated options demos and real imported quotes are explicitly distinguished.')
    {'Options surface':options_view,'Market regimes':regimes_view,'Event backtester':events_view}[selected]()
