package de.uniulm.sp.fe4femo.helper;

public interface LineAnalyser {

    void handleLine(String line);

    LineAnalyser accumulate(LineAnalyser lineAnalyser);
}
