"""Small auditable M15 event engine; paper simulation, not broker execution.

Signal observes a completed candle; market entry uses the NEXT contiguous bar's
open. Source open labels never imply the full candle was known then.
"""
from dataclasses import dataclass,asdict
import math
from collections import deque
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class Risk:
    point_value: float=2.
    tick: float=.25
    commission_rt: float=1.54
    slippage_ticks: float=1.
    max_trade_loss: float=50.
    daily_loss: float=100.
    drawdown: float=500.
    quantity: int=1
    reward_risk: float=2.

    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):raise ValueError('Risk settings must be finite')
        if min(self.point_value,self.tick,self.max_trade_loss,self.daily_loss,self.drawdown,self.reward_risk)<=0 or self.quantity<1 or int(self.quantity)!=self.quantity or min(self.commission_rt,self.slippage_ticks)<0:
            raise ValueError('Invalid risk settings')


@dataclass(frozen=True)
class Event:
    kind: str
    time: pd.Timestamp
    payload: dict


def attribute_to_regimes(result, observations):
    """Use the regime known at SIGNAL candle close, never the later outcome."""
    groups={name:dict(regime=name,proposals=0,trades=0,net=0.) for name in
            ['Lower volatility','Middle volatility','Higher volatility','Unclassified']}
    def known(decision):
        source=pd.Timestamp(decision)-pd.Timedelta(minutes=15)
        return observations.loc[source,'regime'] if source in observations.index else 'Unclassified'
    for e in result['events']:
        if e['event']=='signal':groups[known(e['time'])]['proposals']+=1
    for trade in result['trades']:
        row=groups[known(trade['decision'])];row['trades']+=1;row['net']+=trade['net']
    for row in groups.values():row['expectancy']=row['net']/row['trades'] if row['trades'] else None
    return list(groups.values())


def run(df,risk=Risk(),start=None,signals=None):
    risk.validate()
    if df.empty or df.index.tz is None or not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError('Need nonempty, sorted, unique timezone-aware candles')
    if not np.isfinite(df[['open','high','low','close']]).all().all():
        raise ValueError('Candles must be finite')
    if np.any(np.abs(df[['open','high','low','close']].to_numpy()/risk.tick-np.round(df[['open','high','low','close']].to_numpy()/risk.tick))>1e-6):
        raise ValueError('Prices do not match the configured instrument tick grid')
    # Default example is explicitly an engineering benchmark, not reel logic.
    fast=df.close.ewm(span=10,adjust=False).mean()
    slow=df.close.ewm(span=30,adjust=False).mean()
    side=np.sign(fast-slow)
    proposals=side.where(side.ne(side.shift()),0).fillna(0)
    if signals is not None:proposals=signals.reindex(df.index).fillna(0)
    tr=pd.concat([df.high-df.low,(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean()
    pending=None;position=None;trades=[];events=[];curve=[]
    equity=peak=50000.;day=None;day_start=equity;attempt_day=None
    slip=risk.slippage_ticks*risk.tick;step=pd.Timedelta(minutes=15)
    def event(ts,kind,**kw):events.append(dict(time=str(ts),event=kind,**kw))
    def finish(ts,raw,reason):
        nonlocal position,equity,peak
        p=position
        exit_price=raw if reason=='target' else raw-p['side']*slip
        pnl=p['side']*(exit_price-p['entry'])*risk.point_value*p['quantity']-risk.commission_rt*p['quantity']
        equity+=pnl;peak=max(peak,equity)
        trade=dict(p,exit_time=str(ts),exit=exit_price,reason=reason,net=pnl)
        trades.append(trade);event(ts,'exit',reason=reason,net=pnl,equity=equity);event(ts,'accounting',equity=equity,peak=peak);position=None
    queue=deque()
    for i,(ts,row) in enumerate(df.iterrows()):
        # The open event contains NO high/low/close. Only the later close event
        # exposes the completed candle. Derived signals and orders pass through
        # this same FIFO queue before another source-bar event is introduced.
        queue.append(Event('market_open',ts,dict(open=float(row.open))))
        queue.append(Event('market_close',ts+step,dict(open=float(row.open),high=float(row.high),low=float(row.low),close=float(row.close))))
        while queue:
            message=queue.popleft();now=message.time;data=message.payload
            if message.kind=='signal':
                event(now,'signal',**data)
                queue.append(Event('order',now,data))
                continue
            if message.kind=='order':
                pending=dict(data,decision=str(now))
                event(now,'order',side=data['side'],stop=data['stop'])
                continue
            event(now,message.kind,**data)
            local=ts.tz_convert('America/Chicago');today=local.date();minute=local.hour*60+local.minute
            if message.kind=='market_open':
                opening=data['open']
                if today!=day:
                    if position:finish(now,opening,'data_gap')
                    pending=None;day=today;day_start=equity
                gap=i>0 and ts!=df.index[i-1]+step
                if gap:
                    if pending:event(now,'cancel',reason='missing execution candle')
                    pending=None
                    if position:finish(now,opening,'data_gap')
                if position and minute>=720:finish(now,opening,'session_close')
                if pending and (minute>=720 or minute<510):
                    event(now,'cancel',reason='outside entry session');pending=None
                if pending and position is None:
                    direction=pending['side'];entry=opening+direction*slip
                    stop=pending['stop'];distance=direction*(entry-stop)
                    budgets=[risk.max_trade_loss,risk.daily_loss+equity-day_start,risk.drawdown-(peak-equity)]
                    planned=(distance*risk.point_value+risk.commission_rt+slip*risk.point_value)*risk.quantity
                    if distance>0 and planned<=min(budgets):
                        target=entry+direction*distance*risk.reward_risk
                        target=(math.ceil(target/risk.tick) if direction==1 else math.floor(target/risk.tick))*risk.tick
                        position=dict(side=direction,entry=entry,stop=stop,target=target,quantity=risk.quantity,
                                      decision=pending['decision'],entry_time=str(now),planned_loss=planned)
                        event(now,'fill',price=entry,quantity=risk.quantity)
                    else:event(now,'reject',reason='fill geometry or risk budget',planned_loss=planned)
                    pending=None
                continue
            if position:
                p=position;long=p['side']==1
                stop_hit=data['low']<=p['stop'] if long else data['high']>=p['stop']
                target_hit=data['high']>=p['target']+risk.tick if long else data['low']<=p['target']-risk.tick
                if stop_hit:finish(now,min(data['open'],p['stop']) if long else max(data['open'],p['stop']),'stop')
                elif target_hit:finish(now,p['target'],'target')
            decision_local=now.tz_convert('America/Chicago');dm=decision_local.hour*60+decision_local.minute
            if (position is None and 510<=dm<720 and (start is None or now>=start)
                    and attempt_day!=today and i>=30 and pd.notna(atr.iloc[i]) and proposals.iloc[i]!=0):
                direction=int(proposals.iloc[i])
                if direction not in [-1,1]:raise ValueError('Signal direction must be -1, 0 or +1')
                raw_stop=data['close']-direction*1.5*atr.iloc[i]
                stop=(math.floor(raw_stop/risk.tick) if direction==1 else math.ceil(raw_stop/risk.tick))*risk.tick
                queue.append(Event('signal',now,dict(side=direction,stop=stop)))
                attempt_day=today
            curve.append(dict(time=str(now),equity=equity))
    if position:finish(df.index[-1]+step,df.close.iloc[-1],'end_of_data')
    if curve:curve[-1]['equity']=equity
    if pending:event(df.index[-1]+step,'cancel',reason='end of data')
    pnl=[t['net'] for t in trades];values=[50000.]+[r['equity'] for r in curve]
    dd=float(np.max(np.maximum.accumulate(values)-values))
    return dict(trades=trades,events=events,equity=curve,risk=asdict(risk),
        metrics=dict(trades=len(trades),net=sum(pnl),max_realized_drawdown=dd,
        win_rate=sum(p>0 for p in pnl)/len(pnl) if pnl else None,
        expectancy=sum(pnl)/len(pnl) if pnl else None),
        description='EMA10/30 crossing, 1.5 ATR14 structural stop, next-open market fill, one attempt per Chicago session; engineering benchmark, not a reel strategy')
