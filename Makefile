# GPU Computing SDK Version 4.0.8
all:  
	@$(MAKE) -j8 -C ./shared
	@$(MAKE) -j8 -C ./labs

clean: 
	@$(MAKE) -C ./shared clean
	@$(MAKE) -C ./labs clean

clobber:
	@$(MAKE) -C ./shared clobber
	@$(MAKE) -C ./labs clobber
