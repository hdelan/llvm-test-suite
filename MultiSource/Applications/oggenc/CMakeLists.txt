list(APPEND CFLAGS -fno-strict-aliasing)
list(APPEND LDFLAGS -lm)
set(RUN_OPTIONS -Q -s 901820 - < tune)
llvm_multisource(oggenc)
llvm_test_data(oggenc tune)
