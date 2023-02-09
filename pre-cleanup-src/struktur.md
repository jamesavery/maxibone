src/
    __init__.py
    config/
        constants.py
        paths.py
        threadripper00.json
    lib/
        __init__.py
        cpp/
            cpu/
            cpu_seq/
            gpu/
            best/
            include/
        py/ # TODO tænk over hvordan de vælger implementation -- gerne hvordan det trickler "nedad"
            Istedet for at loade al data ind i ram og så køre blokvist over på GPU, så udnyt async yield til at lave en generator! 
            async memmap! 
            geometry/
                FoR_me.py
    debug-explore/
        *.ipynb
    processing_steps/ # kun cli ting der kører af sig selv (+rapport ting over hvad der skete)
        100-.py
        200-
    pybind/
        *-pybind.cc
    test/
        pybind-*.py
        større-test(s).py
    utils/
        io/
        histograms/
        alternative_processing_steps/
    doitall.sh

sæt ci op som test lokalt > generer fil > github action tjekker om fil rapporten matcher git commit hash og melder korrekt test kørsel (eller noget i den dur!)

under oprydning, hold til samme argument interface som de andre! (i.e. compute_ridges gør ikke ( ͡° ͜ʖ ͡°) )

gennemgå doitall og hiv de relevante ud i processing_steps. Dertil kør alt igennem! 

doitall skal også lave en rapport tex. (tænk applied ML small assignment rapporten)