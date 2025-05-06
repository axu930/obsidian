# Initial overview
Some things copied over from [this](https://fxgears.com/index.php?threads/python-development-environment-jacks-technology-stack.1090/) post regarding the creation of a backtester + live trading algo. 
- It had to be flexible and each symbol I wanted to trade should be able to be dynamically added or removed during run time.
- It had to work on multiple types of securities, not just stocks like I had originally envisioned.
- It had to work with multiple incoming data feeds at the same time.
- It had to work with multiple brokers at the same time. _(You might suspect these two lines relate to latency arb between a fast and slow FX broker, but this had more to do with making the platform broker agnostic.. meaning I wanted to write a strategy once and connect it to any broker I desired instead of being limited by a given broker’s platform or coding environment. In other words: write an MQL4 EA and it only runs on MT4, but write an algo on my platform and it will run seamlessly everywhere. This also ensures that my trade logic is not exposed through the broker's own proprietary platform.)_
- It had to communicate to agents or other app running across a network, that way I can offload some of the trade logic and signal generation workload to more powerful computers when the time comes.
- **Finally, if the platform is going to be this robust, it has to also be able to do many other types of strategies, not just the original market making app.** (That last item was key. I never thought the market making app would be a winner; it was just an excuse to start coding. The big boys of wall street would, in my mind, be faster and smarter at market making than I would ever be.. not to mention they’d have a huge infrastructure advantage as well.)   

